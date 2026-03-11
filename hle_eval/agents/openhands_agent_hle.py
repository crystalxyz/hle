"""
OpenHands CLI agent for HLE evaluation.

Supports two execution modes:
- Local: runs openhands directly via `python -m openhands.core.main` (default)
- Docker: runs each question in an isolated Docker container with `--runtime docker`
"""
import os
import re
import sys
import json
import asyncio
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from openhands_config import OpenHandsConfig
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
    strip_ansi_codes,
)


# ── Docker templates (OpenHands specific) ───────────────────────────────

DOCKERFILE_TEMPLATE = """\
FROM python:3.13-slim

RUN apt-get update && apt-get install -y \\
    bash coreutils git curl build-essential tmux \\
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, install OpenHands from source (PyPI version has e2b dep that fails on arm64)
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@main"

WORKDIR /app

RUN mkdir -p /logs/agent /logs/verifier

# Copy workspace contents (instruction.md, images) to /app
COPY workspace/ ./

# Copy and setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
"""

# Agent-specific command block for the entrypoint
_OPENHANDS_AGENT_COMMAND = """\
# Run OpenHands agent (local runtime, no nested Docker)
# SANDBOX_VOLUMES sets workspace_base to /app (matching Harbor)
SANDBOX_VOLUMES=${PWD}:/workspace:rw python -m openhands.core.main \\
    --task="$(printf '# Task instructions are in /app/instruction.md\\nRead that file for the complete task description.\\n')\""""

ENTRYPOINT_TEMPLATE = build_entrypoint(_OPENHANDS_AGENT_COMMAND)


# ── Local agent ─────────────────────────────────────────────────────────

class OpenHandsHLEAgent:
    """OpenHands agent that uses CLI execution for HLE evaluation."""

    def __init__(self, config: OpenHandsConfig, workspace_root: Path):
        self.config = config
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _build_prompt(
        self,
        question: dict[str, Any],
        workspace: Path,
        image_filename: str | None = None,
    ) -> str:
        response_path = str(workspace.resolve() / "response.txt")
        image_path = str(workspace.resolve() / image_filename) if image_filename else None
        return build_hle_prompt(question, response_path, image_path)

    def _create_config_file(self, workspace: Path) -> Path:
        config_path = workspace.resolve() / "config.toml"
        config_content = f'[llm]\nmodel = "{self.config.model}"\n'
        if self.config.api_key:
            config_content += f'api_key = "{self.config.api_key}"\n'
        if self.config.base_url:
            config_content += f'base_url = "{self.config.base_url}"\n'
        config_path.write_text(config_content)
        return config_path

    async def run_question(
        self,
        question: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            question_id = question["id"]
            workspace = self.workspace_root / f"run_{question_id}"
            workspace.mkdir(parents=True, exist_ok=True)

            # Save image and build instruction
            image_filename = decode_and_save_image(question, workspace)
            instruction_content = self._build_prompt(question, workspace, image_filename)
            instruction_path = workspace / "instruction.md"
            instruction_path.write_text(instruction_content, encoding="utf-8")

            config_path = self._create_config_file(workspace)

            trajectory_dir = workspace.resolve() / "openhands_logs"
            trajectory_dir.mkdir(exist_ok=True)

            cmd = [
                sys.executable, "-m", "openhands.core.main",
                f"--task={instruction_path.read_text()}",
                f"--config-file={config_path}",
            ]

            env = self.config.get_env_dict()
            env["FILE_STORE_PATH"] = str(trajectory_dir)
            env["SAVE_TRAJECTORY_PATH"] = str(trajectory_dir / "trajectory.json")
            env["LLM_LOG_COMPLETIONS_FOLDER"] = str(trajectory_dir / "completions")

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                    env=env,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout,
                )

                stdout_text = stdout.decode("utf-8")
                stderr_text = stderr.decode("utf-8")

                conversation_id = None
                if stdout_text:
                    conv_id_match = re.search(r'Conversation ID: ([a-f0-9-]+)', stdout_text)
                    if conv_id_match:
                        conversation_id = conv_id_match.group(1).replace('-', '')

                save_docker_outputs(workspace, stdout_text, stderr_text)

                if conversation_id:
                    openhands_conv_dir = Path.home() / ".openhands" / "conversations" / conversation_id
                    if openhands_conv_dir.exists():
                        for item in openhands_conv_dir.rglob("*"):
                            if item.is_file():
                                rel_path = item.relative_to(openhands_conv_dir)
                                dest_path = workspace / rel_path
                                dest_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, dest_path)

                metadata = {
                    "question_id": question_id,
                    "returncode": process.returncode,
                    "timeout": False,
                    "error": stderr_text if stderr_text else None,
                    "conversation_id": conversation_id,
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
                    "conversation_id": None,
                }

            except Exception as e:
                metadata = {
                    "question_id": question_id,
                    "returncode": -1,
                    "timeout": False,
                    "error": str(e),
                    "conversation_id": None,
                }

            parsed_answer = parse_answer_file(workspace / "response.txt")
            (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
        semaphore = asyncio.Semaphore(num_workers)
        tasks = [self.run_question(q, semaphore) for q in questions]
        return await tqdm_asyncio.gather(*tasks, desc="Running questions")


# ── Docker agent ────────────────────────────────────────────────────────

class DockerOpenHandsHLEAgent:
    """OpenHands agent that runs entirely inside a per-task Docker container."""

    def __init__(
        self,
        config: OpenHandsConfig,
        workspace_root: Path,
        max_iterations: int | None = None,
    ):
        self.config = config
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.max_iterations = max_iterations

    def _docker_prompt(
        self,
        question: dict[str, Any],
        image_filename: str | None = None,
    ) -> str:
        """Prompt builder for Docker build context."""
        response_path = "/logs/agent/response.txt"
        image_path = f"/app/{image_filename}" if image_filename else None
        return build_hle_prompt(question, response_path, image_path)

    async def run_question(
        self,
        question: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            question_id = question["id"]
            workspace = self.workspace_root / f"run_{question_id}"
            workspace.mkdir(parents=True, exist_ok=True)

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
                    question_id, build_ctx, tag_prefix="hle-openhands",
                )

                docker_cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{logs_dir.resolve()}:/logs",
                    "-v", f"{workspace.resolve()}:/app/host_output",
                ]

                # LLM credentials
                if self.config.api_key:
                    docker_cmd += ["-e", f"LLM_API_KEY={self.config.api_key}"]
                if self.config.base_url:
                    docker_cmd += ["-e", f"LLM_BASE_URL={self.config.base_url}"]
                docker_cmd += ["-e", f"LLM_MODEL={self.config.model}"]

                # OpenHands env vars — matches Harbor openhands.py
                env_vars = {
                    "AGENT_ENABLE_BROWSING": "false",
                    "ENABLE_BROWSER": "false",
                    "SANDBOX_ENABLE_AUTO_LINT": "true",
                    "AGENT_ENABLE_PROMPT_EXTENSIONS": "false",
                    "SKIP_DEPENDENCY_CHECK": "1",
                    "RUN_AS_OPENHANDS": "false",
                    "RUNTIME": "local",
                    "FILE_STORE": "local",
                    "FILE_STORE_PATH": "/logs/agent/",
                    "SAVE_TRAJECTORY_PATH": "/logs/agent/openhands.trajectory.json",
                    "LLM_LOG_COMPLETIONS": "true",
                    "LLM_LOG_COMPLETIONS_FOLDER": "/logs/agent/completions/",
                    "LLM_REASONING_EFFORT": self.config.reasoning_effort,
                }
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
            (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            if task_image:
                await docker_cleanup_task_image(task_image)

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
        if not await docker_verify():
            raise RuntimeError("Docker daemon is not running")
        semaphore = asyncio.Semaphore(num_workers)
        tasks = [self.run_question(q, semaphore) for q in questions]
        return await tqdm_asyncio.gather(*tasks, desc="Running questions (Docker per-task)")


# ── CLI entrypoint ──────────────────────────────────────────────────────

def main(args):
    config = OpenHandsConfig(
        model=args.model,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
    )

    print(f"Configuration: {config}")
    print(f"Runtime: {'docker' if args.docker else 'local'}")

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test").to_dict()
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    if args.task_ids:
        task_ids_set = set(args.task_ids)
        questions = [q for q in questions if q["id"] in task_ids_set]
        print(f"Filtered to {len(questions)} questions matching task IDs: {args.task_ids}")

    if args.sample_rate is not None:
        questions = stratified_sample(questions, args.sample_rate, seed=args.sample_seed)

    if args.max_samples:
        questions = questions[:args.max_samples]

    print(f"Total questions to run: {len(questions)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "openhands-docker" if args.docker else "openhands"
    workspace_name = f"{prefix}_{args.model.replace('/', '_')}_{timestamp}"
    workspace_root = Path(args.output_dir) / workspace_name
    print(f"Workspace: {workspace_root}")

    if args.docker:
        print(f"Max iterations: {args.max_iterations or 'no limit'}")
        agent = DockerOpenHandsHLEAgent(
            config=config,
            workspace_root=workspace_root,
            max_iterations=args.max_iterations,
        )
    else:
        agent = OpenHandsHLEAgent(config=config, workspace_root=workspace_root)

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
        description="Run OpenHands CLI agent on HLE evaluation"
    )
    parser.add_argument("--dataset", type=str, default="cais/hle", help="HLE HuggingFace dataset name")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-5", help="Model name for OpenHands")
    parser.add_argument("--timeout", type=float, default=900.0, help="Timeout in seconds per question (default: 900)")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort level (default: high)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--task_ids", type=str, nargs="+", default=None, help="Specific task IDs to run")
    parser.add_argument("--sample_rate", type=float, default=None, help="Sample rate (0.0-1.0) for stratified sampling")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for stratified sampling (default: 42)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit to first N samples (applied after sampling)")
    parser.add_argument("--output_dir", type=str, default="../jobs", help="Output directory (default: ../jobs)")
    parser.add_argument("--docker", action="store_true", help="Use Docker for isolated agent execution")
    parser.add_argument("--sandbox_image", type=str, default="ghcr.io/all-hands-ai/runtime:0.39-nikolaik", help="Docker sandbox image")
    parser.add_argument("--max_iterations", type=int, default=None, help="Max agent iterations per question in Docker mode")

    args = parser.parse_args()
    if args.sample_rate is not None and not (0.0 <= args.sample_rate <= 1.0):
        parser.error("--sample_rate must be between 0.0 and 1.0")
    main(args)
