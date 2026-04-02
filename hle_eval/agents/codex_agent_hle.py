"""
Codex CLI agent for HLE evaluation.

Supports local and Docker execution (-e/--environment flag).
Uses shared utilities from hle_common for prompt building, image handling,
answer parsing, and Docker lifecycle management.
"""
import os
import json
import shlex
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from codex_config import CodexConfig
from hle_common import (
    build_entrypoint,
    build_hle_prompt,
    create_docker_build_context,
    decode_and_save_image,
    docker_build_task_image,
    docker_cleanup_task_image,
    docker_verify,
    find_docker_response,
    parse_answer_file,
    save_docker_outputs,
    stratified_sample,
)


# ── Docker templates (Codex specific) ─────────────────────────────────

DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y bash coreutils && rm -rf /var/lib/apt/lists/*

# Install packages required by the LLM judge (test_judge.py)
RUN pip install --no-cache-dir openai>=1.59.0 anthropic>=0.40.0 pydantic>=2.10.0

WORKDIR /app

RUN mkdir -p /logs/agent /logs/verifier

# Copy workspace contents (includes images if present) to /app
COPY workspace/ ./

# Copy and setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
"""

# Agent-specific command block for the entrypoint
_CODEX_AGENT_COMMAND = """\
# Run Codex CLI agent
codex exec \\
    -m "$CODEX_MODEL" \\
    -s workspace-write \\
    --json \\
    --color never \\
    --skip-git-repo-check \\
    -- "$(cat /app/instruction.md)" \\
    2>&1 | tee /logs/agent/codex.txt"""

ENTRYPOINT_TEMPLATE = build_entrypoint(_CODEX_AGENT_COMMAND)


# ── Agent class ─────────────────────────────────────────────────────────

class CodexHLEAgent:
    """Codex agent for HLE evaluation (local or Docker)."""

    def __init__(
        self,
        config: CodexConfig,
        workspace_root: Path,
        environment: str = "local",
    ):
        self.config = config
        self.workspace_root = workspace_root
        self.environment = environment
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _build_prompt(
        self,
        question: dict[str, Any],
        workspace: Path,
        image_filename: str | None = None,
    ) -> str:
        if self.environment == "docker":
            response_path = "/logs/agent/response.txt"
            image_path = f"/app/{image_filename}" if image_filename else None
        else:
            response_path = str(workspace / "response.txt")
            image_path = str(workspace / image_filename) if image_filename else None
        return build_hle_prompt(question, response_path, image_path)

    def _docker_prompt(
        self,
        question: dict[str, Any],
        image_filename: str | None = None,
    ) -> str:
        response_path = "/logs/agent/response.txt"
        image_path = f"/app/{image_filename}" if image_filename else None
        return build_hle_prompt(question, response_path, image_path)

    def _build_sandbox_env(self) -> dict[str, str]:
        """Build env var dict for Docker sandboxes."""
        env_vars = {}
        if self.config.api_key:
            env_vars["OPENAI_API_KEY"] = self.config.api_key
        if self.config.base_url:
            env_vars["OPENAI_BASE_URL"] = self.config.base_url
        env_vars["CODEX_MODEL"] = self.config.model
        return env_vars

    # ── Local execution ─────────────────────────────────────────────────

    async def _run_question_local(
        self,
        question: dict[str, Any],
        workspace: Path,
    ) -> dict[str, Any]:
        question_id = question["id"]

        image_filename = decode_and_save_image(question, workspace)
        prompt = self._build_prompt(question, workspace, image_filename)

        cmd = [
            "codex",
            "exec",
            "-m", self.config.model,
            "-s", "workspace-write",
            "--json",
            "--color", "never",
            "--skip-git-repo-check",
            "--",
            prompt,
        ]

        try:
            env = self.config.get_env_dict()

            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            shell_cmd = f"cd {shlex.quote(str(workspace))} && {cmd_str} 2>&1 | tee codex_trajectory.json"

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

            metadata = {
                "question_id": question_id,
                "returncode": process.returncode,
                "timeout": False,
                "error": stderr.decode("utf-8") if stderr else None,
            }

        except asyncio.TimeoutError:
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

        parsed_answer = parse_answer_file(workspace / "response.txt")

        (workspace / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        return {
            "question_id": question_id,
            "response": parsed_answer.get("raw", "") if parsed_answer else None,
            "parsed": parsed_answer,
            "metadata": metadata,
            "workspace": str(workspace),
        }

    # ── Docker execution ────────────────────────────────────────────────

    async def _run_question_docker(
        self,
        question: dict[str, Any],
        workspace: Path,
    ) -> dict[str, Any]:
        question_id = question["id"]

        logs_dir = workspace / "logs"
        logs_dir.mkdir(exist_ok=True)

        task_image = None
        try:
            build_ctx, _ = create_docker_build_context(
                question, workspace,
                dockerfile=DOCKERFILE_TEMPLATE,
                entrypoint=ENTRYPOINT_TEMPLATE,
                build_prompt=self._docker_prompt,
            )
            task_image = await docker_build_task_image(
                question_id, build_ctx, tag_prefix="hle-codex",
            )

            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{logs_dir.resolve()}:/logs",
                "-v", f"{workspace.resolve()}:/app/host_output",
            ]

            env_vars = self._build_sandbox_env()
            for key, val in env_vars.items():
                docker_cmd += ["-e", f"{key}={val}"]

            docker_cmd.append(task_image)

            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout,
            )

            stdout_text = stdout.decode("utf-8")
            stderr_text = stderr.decode("utf-8")
            save_docker_outputs(workspace, stdout_text, stderr_text)

            metadata = {
                "question_id": question_id,
                "returncode": process.returncode,
                "timeout": False,
                "error": stderr_text if stderr_text else None,
                "runtime": "docker",
                "task_image": task_image,
            }

        except asyncio.TimeoutError:
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
                "runtime": "docker",
                "task_image": task_image,
            }

        except Exception as e:
            metadata = {
                "question_id": question_id,
                "returncode": -1,
                "timeout": False,
                "error": str(e),
                "runtime": "docker",
                "task_image": task_image if task_image else "build_failed",
            }

        parsed_answer = find_docker_response(logs_dir, workspace)

        (workspace / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        if task_image:
            await docker_cleanup_task_image(task_image)

        return {
            "question_id": question_id,
            "response": parsed_answer.get("raw", "") if parsed_answer else None,
            "parsed": parsed_answer,
            "metadata": metadata,
            "workspace": str(workspace),
        }

    # ── Orchestration ───────────────────────────────────────────────────

    async def run_question(
        self,
        question: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            workspace = self.workspace_root / f"run_{question['id']}"
            workspace.mkdir(parents=True, exist_ok=True)

            if self.environment == "docker":
                return await self._run_question_docker(question, workspace)
            return await self._run_question_local(question, workspace)

    async def run_all_questions(
        self,
        questions: list[dict[str, Any]],
        num_workers: int = 4,
    ) -> list[dict[str, Any]]:
        if self.environment == "docker":
            if not await docker_verify():
                raise RuntimeError("Docker daemon is not running")

        semaphore = asyncio.Semaphore(num_workers)
        tasks = [self.run_question(q, semaphore) for q in questions]
        desc = f"Running questions ({self.environment})"
        return await tqdm_asyncio.gather(*tasks, desc=desc)


# ── CLI entrypoint ──────────────────────────────────────────────────────

def main(args):
    config = CodexConfig(model=args.model, timeout=args.timeout)

    print(f"Configuration: {config}")
    print(f"Environment: {args.environment}")

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test").to_dict()
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # Collect task IDs from --task_ids and/or --task_ids_file
    task_ids_set = set(args.task_ids) if args.task_ids else set()
    if args.task_ids_file:
        file_ids = json.loads(Path(args.task_ids_file).read_text())
        task_ids_set.update(file_ids)
    if task_ids_set:
        questions = [q for q in questions if q["id"] in task_ids_set]
        print(f"Filtered to {len(questions)} questions matching {len(task_ids_set)} task IDs")

    if args.sample_rate is not None:
        questions = stratified_sample(questions, args.sample_rate, seed=args.sample_seed)

    if args.max_samples:
        questions = questions[:args.max_samples]

    print(f"Total questions to run: {len(questions)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_label = {"local": "codex", "docker": "codex-docker"}
    prefix = env_label.get(args.environment, f"codex-{args.environment}")
    workspace_name = f"{prefix}_{args.model}_{timestamp}"
    workspace_root = (Path(args.output_dir) / workspace_name).resolve()
    print(f"Workspace: {workspace_root}")

    agent = CodexHLEAgent(
        config=config,
        workspace_root=workspace_root,
        environment=args.environment,
    )
    results = asyncio.run(agent.run_all_questions(questions, num_workers=args.num_workers))

    output_file = workspace_root / "results.json"
    predictions = {}
    for result in results:
        predictions[result["question_id"]] = {
            "model": args.model,
            "response": result["response"],
            "parsed": result["parsed"],
            "metadata": result["metadata"],
            "workspace": result["workspace"],
        }
    output_file.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_file}")

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
        description="Run Codex CLI agent on HLE evaluation"
    )
    parser.add_argument("--dataset", type=str, default="cais/hle", help="HLE HuggingFace dataset name")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name for codex exec")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout in seconds per question (default: 1200)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--task_ids", type=str, nargs="+", default=None, help="Specific task IDs to run")
    parser.add_argument("--task_ids_file", type=str, default=None, help="JSON file containing a list of task IDs to run")
    parser.add_argument("--sample_rate", type=float, default=None, help="Sample rate (0.0-1.0) for stratified sampling")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for stratified sampling (default: 42)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit to first N samples (applied after sampling)")
    parser.add_argument("--output_dir", type=str, default="../jobs", help="Output directory (default: ../jobs)")
    parser.add_argument(
        "-e", "--environment",
        choices=["local", "docker"],
        default="local",
        help="Execution environment: local (default) or docker",
    )

    args = parser.parse_args()
    if args.sample_rate is not None and not (0.0 <= args.sample_rate <= 1.0):
        parser.error("--sample_rate must be between 0.0 and 1.0")
    main(args)
