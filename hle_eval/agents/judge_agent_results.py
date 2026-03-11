"""
Judge agent results with retry logic.

Output is a clean report categorized into pass/fail/no_response,
containing only judge verdicts and reasoning (no raw data).
"""
import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset


# Answer overrides for tasks with incorrect/incomplete answers in the HLE dataset
ANSWER_OVERRIDES: dict[str, str] = {
    "6713a4c60223609143188d32": (
        "Product A: Methyl 5-(2-acetamidoethyl)-2,3-dihydro-1H-pyrrolizine-6-carboxylate\n"
        "Product B: 7a-(2-Oxopyrrolidine-1-carbonyl)-5,6,7,7a-tetrahydro-3H-pyrrolizin-3-one\n"
        "Product C: 1-(Acetylprolyl)pyrrolidin-2-one"
    ),
}

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:


[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]  # 100% reliability


async def extract_answer_with_retry(
    client: AsyncOpenAI,
    judge_model: str,
    question: str,
    correct_answer: str,
    response: str,
    max_retries: int = 20,
) -> dict | None:
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response
    )

    for attempt in range(max_retries):
        try:
            api_response = await client.beta.chat.completions.parse(
                model=judge_model,
                max_completion_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                response_format=ExtractedAnswer,
            )
            content = api_response.choices[0].message.parsed
            return {
                "correct_answer": correct_answer,
                "model_answer": content.extracted_final_answer,
                "reasoning": content.reasoning,
                "correct": content.correct,
                "confidence": content.confidence,
            }

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} retries: {e}")
                return None
            wait_time = min(2 ** attempt, 16)
            await asyncio.sleep(wait_time)

    return None


async def judge_question(
    client: AsyncOpenAI,
    judge_model: str,
    question: dict,
    predictions: dict,
    max_retries: int = 20,
) -> tuple[str | None, dict | None]:
    unique_id = question["id"]

    if unique_id not in predictions:
        return None, None

    prediction = copy.deepcopy(predictions[unique_id])

    # Skip if already judged
    if "judge_response" in prediction:
        return unique_id, prediction

    # No response available
    if not prediction.get("response"):
        return unique_id, None

    question_text = question["question"]
    correct_answer = ANSWER_OVERRIDES.get(unique_id, question["answer"])
    response = prediction["response"]

    content = await extract_answer_with_retry(
        client=client,
        judge_model=judge_model,
        question=question_text,
        correct_answer=correct_answer,
        response=response,
        max_retries=max_retries,
    )

    if content is not None:
        prediction["judge_response"] = content
        return unique_id, prediction
    else:
        return unique_id, None


async def judge_all_responses(
    client: AsyncOpenAI,
    judge_model: str,
    questions: list[dict],
    predictions: dict,
    num_workers: int,
    max_retries: int = 20,
) -> list[tuple[str | None, dict | None]]:
    semaphore = asyncio.Semaphore(num_workers)

    async def bounded_judge(question):
        async with semaphore:
            return await judge_question(
                client=client,
                judge_model=judge_model,
                question=question,
                predictions=predictions,
                max_retries=max_retries,
            )

    tasks = [bounded_judge(q) for q in questions]
    results = await tqdm_asyncio.gather(*tasks, desc="Judging responses")
    return results


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p='2', beta=100):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]

    if not bins:
        return 0.0

    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def build_report(
    judged_predictions: dict,
    predictions: dict,
    question_lookup: dict[str, dict],
    judge_model: str,
) -> dict:
    """Build a clean report categorized into pass/fail/no_response."""
    passed = []
    failed = []
    no_response = []

    for qid in predictions:
        question = question_lookup.get(qid, {})
        category = question.get("category", "unknown")

        # Task had no response (timeout, error, or missing)
        if qid not in judged_predictions or judged_predictions[qid] is None:
            pred = predictions[qid]
            error = None
            if isinstance(pred, dict):
                meta = pred.get("metadata", {})
                if meta.get("timeout"):
                    error = "timeout"
                elif meta.get("error"):
                    error = str(meta["error"])[:200]
                elif not pred.get("response"):
                    error = "no response generated"
            no_response.append({
                "question_id": qid,
                "category": category,
                "error": error,
            })
            continue

        jp = judged_predictions[qid]
        judge = jp.get("judge_response")
        if not judge:
            no_response.append({
                "question_id": qid,
                "category": category,
                "error": "judge returned no result",
            })
            continue

        entry = {
            "question_id": qid,
            "category": category,
            "correct": judge["correct"],
            "model_answer": judge["model_answer"],
            "correct_answer": judge["correct_answer"],
            "reasoning": judge["reasoning"],
            "confidence": judge["confidence"],
        }

        if judge["correct"] == "yes":
            passed.append(entry)
        else:
            failed.append(entry)

    # Compute metrics
    n = len(predictions)
    n_pass = len(passed)
    n_fail = len(failed)
    n_no_response = len(no_response)
    accuracy = round(100 * n_pass / n, 2) if n > 0 else 0.0
    ci = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2) if n > 0 else 0.0

    correct_arr = np.array([1] * n_pass + [0] * n_fail)
    confidence_arr = np.array(
        [e["confidence"] for e in passed] + [e["confidence"] for e in failed]
    ) / 100
    cal_err = round(100 * calib_err(confidence_arr, correct_arr, p='2', beta=100), 2) if len(correct_arr) > 0 else 0.0

    # Category breakdown
    cat_counts: dict[str, dict[str, int]] = {}
    for e in passed:
        cat_counts.setdefault(e["category"], {"pass": 0, "fail": 0, "no_response": 0})["pass"] += 1
    for e in failed:
        cat_counts.setdefault(e["category"], {"pass": 0, "fail": 0, "no_response": 0})["fail"] += 1
    for e in no_response:
        cat_counts.setdefault(e["category"], {"pass": 0, "fail": 0, "no_response": 0})["no_response"] += 1

    return {
        "summary": {
            "judge_model": judge_model,
            "total": n,
            "pass": n_pass,
            "fail": n_fail,
            "no_response": n_no_response,
            "accuracy": accuracy,
            "accuracy_ci": f"+/- {ci}%",
            "calibration_error": cal_err,
            "by_category": dict(sorted(cat_counts.items())),
        },
        "pass": sorted(passed, key=lambda x: x["question_id"]),
        "fail": sorted(failed, key=lambda x: x["question_id"]),
        "no_response": sorted(no_response, key=lambda x: x["question_id"]),
    }


def load_results_from_workspace(workspace_path: str) -> dict:
    workspace = Path(workspace_path)
    results_file = workspace / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No results.json found in {workspace_path}")
    with open(results_file, "r") as f:
        return json.load(f)


def main(args):
    num_workers = args.num_workers
    max_retries = 20

    # Load results
    if args.workspace:
        print(f"Loading results from workspace: {args.workspace}")
        predictions = load_results_from_workspace(args.workspace)
        output_filepath = Path(args.workspace) / "judged_results.json"
    elif args.predictions:
        print(f"Loading predictions from file: {args.predictions}")
        with open(args.predictions, "r") as f:
            predictions = json.load(f)
        output_filepath = Path(f"judged_{os.path.basename(args.predictions)}")
    else:
        raise ValueError("Must provide either --workspace or --predictions")

    client = AsyncOpenAI(timeout=300.0, max_retries=0)

    print(f"Judge model: {args.judge}")

    # Load dataset
    dataset = load_dataset("cais/hle", split="test").to_dict()
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    question_lookup = {q["id"]: q for q in questions}

    # Load existing judged results (for incremental judging)
    judged_predictions = {}
    raw_judged_path = output_filepath.parent / ".judged_raw.json" if args.workspace else Path(f".judged_raw_{os.path.basename(args.predictions)}.json")
    if os.path.exists(raw_judged_path):
        with open(raw_judged_path, "r") as f:
            judged_predictions = json.load(f)

    # Filter to unjudged questions
    questions_to_judge = [
        q for q in questions
        if q["id"] in predictions and q["id"] not in judged_predictions
    ]

    # Also track tasks with no response (don't need judging)
    no_response_ids = set()
    for qid, pred in predictions.items():
        if not pred.get("response"):
            no_response_ids.add(qid)

    questions_to_judge = [q for q in questions_to_judge if q["id"] not in no_response_ids]

    print(f"Total tasks: {len(predictions)}")
    print(f"Already judged: {len(judged_predictions)}")
    print(f"No response (skip): {len(no_response_ids)}")
    print(f"To judge: {len(questions_to_judge)}")

    if questions_to_judge:
        results = asyncio.run(
            judge_all_responses(
                client=client,
                judge_model=args.judge,
                questions=questions_to_judge,
                predictions=predictions,
                num_workers=num_workers,
                max_retries=max_retries,
            )
        )

        for unique_id, prediction in results:
            if unique_id is not None:
                judged_predictions[unique_id] = prediction

        # Save raw judged data (for incremental re-runs)
        with open(raw_judged_path, "w") as f:
            json.dump(judged_predictions, f, indent=2)

    # Build and save clean report
    report = build_report(judged_predictions, predictions, question_lookup, args.judge)

    print(f"\nSaving report to: {output_filepath}")
    with open(output_filepath, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    s = report["summary"]
    print(f"\n{'=' * 50}")
    print(f"  Judge: {s['judge_model']}")
    print(f"  Total: {s['total']}")
    print(f"  Pass:  {s['pass']}  |  Fail: {s['fail']}  |  No Response: {s['no_response']}")
    print(f"  Accuracy: {s['accuracy']}% {s['accuracy_ci']}")
    print(f"  Calibration Error: {s['calibration_error']}")
    print(f"{'=' * 50}")
    if s["by_category"]:
        print(f"\n  {'Category':<30} {'Pass':>5} {'Fail':>5} {'N/A':>5}")
        print(f"  {'-'*50}")
        for cat, counts in s["by_category"].items():
            print(f"  {cat:<30} {counts['pass']:>5} {counts['fail']:>5} {counts['no_response']:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge agent results and produce a clean pass/fail/no_response report"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--workspace", type=str,
        help="Path to workspace directory (e.g., jobs/claude-code_haiku_20260309/)",
    )
    input_group.add_argument(
        "--predictions", type=str,
        help="Path to predictions JSON file",
    )

    parser.add_argument(
        "--judge", type=str, default="gpt-5",
        help="Judge model name (default: gpt-5)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=10,
        help="Number of concurrent workers (default: 10)",
    )

    args = parser.parse_args()
    main(args)
