"""
Claude Code CLI agent for HLE evaluation, based on LAB-Bench codex_agent.py patterns.

This script uses the actual `claude` CLI command to evaluate HLE questions,
with clean workspace logging and answer parsing.
"""
import os
import re
import json
import shlex
import asyncio
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Any
from collections import defaultdict

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from claude_config import ClaudeConfig


class ClaudeHLEAgent:
    """Claude Code agent that uses CLI execution for HLE evaluation."""

    def __init__(
        self,
        config: ClaudeConfig,
        workspace_root: Path,
    ):
        """
        Initialize Claude HLE agent.

        Args:
            config: Claude configuration with API credentials
            workspace_root: Root directory for storing run artifacts
        """
        self.config = config
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _build_prompt(self, question: dict[str, Any]) -> str:
        """
        Build the prompt for a question.

        Args:
            question: Question dict with 'id', 'question', 'image', etc.

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are answering a challenging academic question from Humanity's Last Exam.

Instruction:
- Provide your response in the following format:
  Explanation: {{your explanation for your answer choice}}
  Answer: {{your chosen answer}}
  Confidence: {{your confidence score between 0% and 100% for your answer}}
- Write your response to `answer.txt`.

Question:
{question['question']}
"""

        # Add image reference if present
        if question.get('image'):
            prompt += f"\nImage: See the image file at `image.png`\n"

        return prompt

    def _build_cli_command(
        self,
        workspace: Path,
    ) -> list[str]:
        """
        Build claude CLI command following LAB-Bench pattern.

        Args:
            workspace: Workspace directory path

        Returns:
            Command list for subprocess execution
        """
        # Claude Code command that reads prompt.txt and executes
        prompt_instruction = "Read the file `prompt.txt` and follow the instructions to answer the question. If there is an image file, make sure to view it first."

        return [
            "claude",
            "-p", prompt_instruction,
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.config.model,
            "--dangerously-skip-permissions",
        ]

    def _parse_answer_file(self, answer_file: Path) -> dict[str, str] | None:
        """
        Parse the answer.txt file for Explanation, Answer, and Confidence.

        Args:
            answer_file: Path to answer.txt file

        Returns:
            Dict with 'explanation', 'answer', 'confidence' or None if parsing fails
        """
        if not answer_file.exists():
            return None

        content = answer_file.read_text(encoding="utf-8")

        # Try to extract structured response
        explanation_match = re.search(
            r"Explanation:\s*(.+?)(?=Answer:|Confidence:|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        answer_match = re.search(
            r"Answer:\s*(.+?)(?=Confidence:|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        confidence_match = re.search(
            r"Confidence:\s*(.+?)(?=$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )

        if answer_match:
            return {
                "explanation": explanation_match.group(1).strip() if explanation_match else "",
                "answer": answer_match.group(1).strip(),
                "confidence": confidence_match.group(1).strip() if confidence_match else "",
                "raw": content,
            }

        # Fallback: return raw content if structured parsing fails
        return {
            "explanation": "",
            "answer": content.strip(),
            "confidence": "",
            "raw": content,
        }

    async def run_question(
        self,
        question: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """
        Run a single question through claude CLI.

        Args:
            question: Question dict from HLE dataset
            semaphore: Semaphore for concurrency control

        Returns:
            Result dict with question_id, response, and metadata
        """
        async with semaphore:
            question_id = question["id"]

            # Create workspace directory for this question
            workspace = self.workspace_root / f"run_{question_id}"
            workspace.mkdir(parents=True, exist_ok=True)

            # Save image if present
            if question.get("image"):
                image_path = workspace / "image.png"
                # HLE images are base64 encoded data URLs
                if question["image"].startswith("data:"):
                    import base64
                    # Extract base64 data
                    header, encoded = question["image"].split(",", 1)
                    image_data = base64.b64decode(encoded)
                    image_path.write_bytes(image_data)

            # Build prompt and save it
            prompt = self._build_prompt(question)
            prompt_file = workspace / "prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")

            # Build and execute CLI command
            cmd = self._build_cli_command(workspace)
            trajectory_file = workspace / "claude_trajectory.jsonl"

            # Execute with timeout
            try:
                # Change to workspace directory for execution
                env = self.config.get_env_dict()

                # Use shell command with tee for streaming output capture
                cmd_str = " ".join(shlex.quote(c) for c in cmd)
                # Use just the filename since we cd into the workspace
                shell_cmd = f"cd {shlex.quote(str(workspace))} && {cmd_str} 2>&1 | tee claude_trajectory.jsonl"

                process = await asyncio.create_subprocess_shell(
                    shell_cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout,
                )

                returncode = process.returncode

                # Save metadata
                metadata = {
                    "question_id": question_id,
                    "returncode": returncode,
                    "timeout": False,
                    "error": stderr.decode("utf-8") if stderr else None,
                }

            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass

                metadata = {
                    "question_id": question_id,
                    "returncode": -1,
                    "timeout": True,
                    "error": f"Execution timed out after {self.config.timeout}s",
                }

            except Exception as e:
                metadata = {
                    "question_id": question_id,
                    "returncode": -1,
                    "timeout": False,
                    "error": str(e),
                }

            # Parse answer file
            answer_file = workspace / "answer.txt"
            parsed_answer = self._parse_answer_file(answer_file)

            # Save metadata to workspace
            metadata_file = workspace / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            return {
                "question_id": question_id,
                "response": parsed_answer.get("raw", "") if parsed_answer else None,
                "parsed": parsed_answer,
                "metadata": metadata,
                "workspace": str(workspace),
            }

    async def run_all_questions(
        self,
        questions: list[dict[str, Any]],
        num_workers: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Run all questions with concurrent execution.

        Args:
            questions: List of question dicts from HLE dataset
            num_workers: Number of concurrent workers (default: 4)

        Returns:
            List of result dicts
        """
        semaphore = asyncio.Semaphore(num_workers)

        tasks = [self.run_question(q, semaphore) for q in questions]
        results = await tqdm_asyncio.gather(*tasks, desc="Running questions")

        return results


def stratified_sample(questions: list[dict[str, Any]], sample_rate: float, seed: int = 42) -> list[dict[str, Any]]:
    """
    Sample from each category at the given rate using specified seed.

    Args:
        questions: List of question dicts from HLE dataset
        sample_rate: Sample rate (0.0-1.0) to apply to each category
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Sampled list of questions
    """
    random.seed(seed)

    # Group questions by category
    category_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        category = question.get("category", "unknown")
        category_questions[category].append(question)

    # Sample from each category
    sampled_questions: list[dict[str, Any]] = []
    print(f"\n=== Stratified Sampling (rate={sample_rate}, seed={seed}) ===")
    for category in sorted(category_questions.keys()):
        questions_in_category = category_questions[category]
        n_samples = max(1, round(len(questions_in_category) * sample_rate))
        sampled = random.sample(questions_in_category, min(n_samples, len(questions_in_category)))
        sampled_questions.extend(sampled)
        print(f"  Category '{category}': {len(sampled)}/{len(questions_in_category)} questions sampled")

    print(f"  Total: {len(sampled_questions)} questions sampled from {len(questions)}")
    print("=" * 60 + "\n")

    return sampled_questions


def main(args):
    """Main execution function."""
    # Create config
    config = ClaudeConfig(
        model=args.model,
        timeout=args.timeout,
    )

    print(f"Configuration: {config}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test").to_dict()

    # Convert to list of dicts
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # Apply stratified sampling if sample_rate is specified
    if args.sample_rate is not None:
        questions = stratified_sample(questions, args.sample_rate, seed=args.sample_seed)

    # Limit samples if specified (after sampling)
    if args.max_samples:
        questions = questions[:args.max_samples]

    print(f"Total questions to run: {len(questions)}")

    # Create workspace directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"claude-code_{args.model}_{timestamp}"
    workspace_root = Path(args.output_dir) / workspace_name

    print(f"Workspace: {workspace_root}")

    # Create agent
    agent = ClaudeHLEAgent(
        config=config,
        workspace_root=workspace_root,
    )

    # Run evaluation
    results = asyncio.run(agent.run_all_questions(questions, num_workers=args.num_workers))

    # Save aggregated results
    output_file = workspace_root / "results.json"
    predictions = {}

    for result in results:
        question_id = result["question_id"]
        predictions[question_id] = {
            "model": args.model,
            "response": result["response"],
            "parsed": result["parsed"],
            "metadata": result["metadata"],
            "workspace": result["workspace"],
        }

    output_file.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    total = len(results)
    successful = sum(1 for r in results if r["response"] is not None)
    timed_out = sum(1 for r in results if r["metadata"]["timeout"])
    errors = sum(1 for r in results if r["metadata"]["returncode"] != 0 and not r["metadata"]["timeout"])

    print("\n=== Summary ===")
    print(f"Total questions: {total}")
    print(f"Successful: {successful}")
    print(f"Timed out: {timed_out}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Claude Code CLI agent on HLE evaluation (LAB-Bench style)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cais/hle",
        help="HLE HuggingFace dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sonnet",
        help="Model name for claude CLI (sonnet, opus, haiku)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for each question (default: 600)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4)",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=None,
        help="Sample rate (0.0-1.0) for stratified sampling by category (default: None = no sampling)",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (applied after sampling, for testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../jobs",
        help="Output directory for results (default: ../jobs)",
    )

    args = parser.parse_args()

    # Validate sample_rate range
    if args.sample_rate is not None and not (0.0 <= args.sample_rate <= 1.0):
        parser.error("--sample_rate must be between 0.0 and 1.0")

    main(args)
