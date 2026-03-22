"""
Claude Code CLI agent for HLE evaluation.

Supports local, Docker, and Daytona execution (-e/--environment flag).
"""
import os
import json
import shlex
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any
from uuid import uuid4

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from claude_config import ClaudeConfig
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

# Optional Daytona SDK
try:
    from daytona import (
        AsyncDaytona,
        CreateSandboxFromImageParams,
        Image,
        SessionExecuteRequest,
    )
    HAS_DAYTONA = True
except ImportError:
    HAS_DAYTONA = False


# ── Docker templates (Claude Code specific) ─────────────────────────────

DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y bash coreutils && rm -rf /var/lib/apt/lists/*

# Install packages required by the LLM judge (test_judge.py)
# Supports both OpenAI and Anthropic/Claude models as judges
RUN pip install --no-cache-dir openai>=1.59.0 anthropic>=0.40.0 pydantic>=2.10.0

WORKDIR /app

RUN mkdir -p /logs/agent /logs/verifier

# Copy workspace contents (includes images if present) to /app
# This allows vision-capable models to directly see image files in their working directory
COPY workspace/ ./

# Copy and setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
"""

# Agent-specific command block for the entrypoint
_CLAUDE_AGENT_COMMAND = """\
# Setup Claude config directory (matches Harbor)
mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app \\
    $CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig \\
    $CLAUDE_CONFIG_DIR/todos && \\
if [ -d ~/.claude/skills ]; then \\
    cp -r ~/.claude/skills $CLAUDE_CONFIG_DIR/skills 2>/dev/null || true; \\
fi

# Run Claude Code CLI agent — read task from instruction.md (matches Harbor CLI flags)
claude --verbose --output-format=stream-json \\
    --permission-mode bypassPermissions \\
    --print \\
    -- "# Task instructions are in /app/instruction.md\\nRead that file for the complete task description." \\
    2>&1 </dev/null | tee /logs/agent/claude-code.txt"""

ENTRYPOINT_TEMPLATE = build_entrypoint(_CLAUDE_AGENT_COMMAND)


# ── Claude Code trajectory parsing ──────────────────────────────────────

def parse_claude_trajectory(trajectory_file: Path) -> list[dict[str, Any]]:
    """Parse a Claude Code stream-json trajectory file into structured steps."""
    if not trajectory_file.exists():
        return []

    steps = []
    try:
        content = trajectory_file.read_text(encoding="utf-8")
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")

            if msg_type == "system":
                steps.append({
                    "role": "system",
                    "type": obj.get("subtype", "init"),
                    "model": obj.get("model"),
                    "tools": obj.get("tools"),
                })

            elif msg_type == "assistant":
                message = obj.get("message", {})
                for block in message.get("content", []):
                    block_type = block.get("type", "")
                    if block_type == "thinking":
                        steps.append({
                            "role": "assistant",
                            "type": "thinking",
                            "content": block.get("thinking", ""),
                        })
                    elif block_type == "text":
                        steps.append({
                            "role": "assistant",
                            "type": "text",
                            "content": block.get("text", ""),
                        })
                    elif block_type == "tool_use":
                        steps.append({
                            "role": "assistant",
                            "type": "tool_call",
                            "tool_name": block.get("name", ""),
                            "tool_id": block.get("id", ""),
                            "input": block.get("input", {}),
                        })

            elif msg_type == "user":
                message = obj.get("message", {})
                for block in message.get("content", []):
                    if block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            parts = []
                            for part in tool_content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    parts.append(part.get("text", ""))
                                elif isinstance(part, str):
                                    parts.append(part)
                            tool_content = "\n".join(parts)
                        steps.append({
                            "role": "tool",
                            "type": "observation",
                            "tool_use_id": block.get("tool_use_id", ""),
                            "content": tool_content,
                            "is_error": block.get("is_error", False),
                        })

            elif msg_type == "result":
                steps.append({
                    "role": "system",
                    "type": "result",
                    "subtype": obj.get("subtype", ""),
                    "cost_usd": obj.get("cost_usd"),
                    "duration_ms": obj.get("duration_ms"),
                    "duration_api_ms": obj.get("duration_api_ms"),
                    "num_turns": obj.get("num_turns"),
                    "input_tokens": obj.get("usage", {}).get("input_tokens"),
                    "output_tokens": obj.get("usage", {}).get("output_tokens"),
                })
    except Exception:
        pass

    return steps


# ── Agent class ─────────────────────────────────────────────────────────

class ClaudeHLEAgent:
    """Claude Code agent for HLE evaluation (local, Docker, or Daytona)."""

    def __init__(
        self,
        config: ClaudeConfig,
        workspace_root: Path,
        environment: str = "local",
    ):
        self.config = config
        self.workspace_root = workspace_root
        self.environment = environment  # "local", "docker", "daytona"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._daytona: "AsyncDaytona | None" = None

    def _build_prompt(
        self,
        question: dict[str, Any],
        workspace: Path,
        image_filename: str | None = None,
    ) -> str:
        if self.environment in ("docker", "daytona"):
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
        """Prompt builder for Docker build context (no workspace arg)."""
        response_path = "/logs/agent/response.txt"
        image_path = f"/app/{image_filename}" if image_filename else None
        return build_hle_prompt(question, response_path, image_path)

    def _build_harbor_env(self) -> dict[str, str]:
        """Build the Harbor-matching env var dict."""
        env_vars = {
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
            "ANTHROPIC_MODEL": self.config.model,
        }

        if self.config.base_url:
            env_vars["ANTHROPIC_DEFAULT_SONNET_MODEL"] = self.config.model
            env_vars["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self.config.model
            env_vars["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = self.config.model
            env_vars["CLAUDE_CODE_SUBAGENT_MODEL"] = self.config.model

        max_thinking = os.environ.get("MAX_THINKING_TOKENS")
        if max_thinking:
            env_vars["MAX_THINKING_TOKENS"] = max_thinking

        max_output = os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        if max_output:
            env_vars["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = max_output

        return env_vars

    def _build_sandbox_env(self) -> dict[str, str]:
        """Build the full env var dict for Docker/Daytona sandboxes."""
        env_vars = self._build_harbor_env()
        env_vars["CLAUDE_CONFIG_DIR"] = "/logs/agent/sessions"

        if self.config.api_key:
            env_vars["ANTHROPIC_API_KEY"] = self.config.api_key
        if self.config.base_url:
            env_vars["ANTHROPIC_BASE_URL"] = self.config.base_url

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
            "claude",
            "-p", prompt,
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", "bypassPermissions",
        ]

        try:
            env = self.config.get_env_dict()
            env.update(self._build_harbor_env())

            cmd_str = " ".join(shlex.quote(c) for c in cmd)
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

        raw_trajectory_file = workspace / "claude_trajectory.jsonl"
        trajectory = parse_claude_trajectory(raw_trajectory_file)
        raw_trajectory_file.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

        (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "question_id": question_id,
            "response": parsed_answer.get("raw", "") if parsed_answer else None,
            "parsed": parsed_answer,
            "trajectory": trajectory,
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
                question_id, build_ctx, tag_prefix="hle-claude",
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

        raw_trajectory_file = logs_dir / "agent" / "claude-code.txt"
        trajectory = parse_claude_trajectory(raw_trajectory_file)
        if raw_trajectory_file.exists():
            raw_trajectory_file.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

        (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if task_image:
            await docker_cleanup_task_image(task_image)

        return {
            "question_id": question_id,
            "response": parsed_answer.get("raw", "") if parsed_answer else None,
            "parsed": parsed_answer,
            "trajectory": trajectory,
            "metadata": metadata,
            "workspace": str(workspace),
        }

    # ── Daytona execution ───────────────────────────────────────────────

    async def _daytona_exec(
        self,
        sandbox: Any,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Execute a command in a Daytona sandbox using session-based async polling.

        Matches Harbor's exec pattern: bash -lc wrapper, iterative env prefix,
        shell timeout command, and session-based polling.
        """
        session_id = str(uuid4())
        await sandbox.process.create_session(session_id)

        # Build full command matching Harbor's exec() pattern:
        # 1. Wrap in bash -lc
        full_cmd = f"bash -lc {shlex.quote(command)}"
        # 2. Prepend env vars iteratively (matches Harbor)
        if env:
            for key, value in env.items():
                full_cmd = f"{key}={shlex.quote(value)} {full_cmd}"
        # 3. Wrap with shell timeout (Harbor uses both shell + SDK timeout)
        #    Use bash -c to ensure env vars are treated as assignments, not commands
        if timeout:
            full_cmd = f"timeout {timeout} bash -c {shlex.quote(full_cmd)}"
        # 4. Prepend cwd
        if cwd:
            full_cmd = f"cd {cwd} && {full_cmd}"

        response = await sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(command=full_cmd, run_async=True),
            timeout=timeout,
        )

        if response.cmd_id is None:
            raise RuntimeError("Cannot find command ID.")

        # Poll for completion (1s interval matches Harbor)
        poll_timeout = (timeout or 1200) + 60  # allow some buffer beyond command timeout
        poll_start = asyncio.get_event_loop().time()
        cmd = await sandbox.process.get_session_command(session_id, response.cmd_id)
        while cmd.exit_code is None:
            elapsed = asyncio.get_event_loop().time() - poll_start
            if elapsed > poll_timeout:
                raise asyncio.TimeoutError(
                    f"Daytona command polling timed out after {int(elapsed)}s"
                )
            await asyncio.sleep(1)
            cmd = await sandbox.process.get_session_command(session_id, response.cmd_id)

        logs = await sandbox.process.get_session_command_logs(session_id, response.cmd_id)
        return logs.stdout or "", logs.stderr or "", int(cmd.exit_code)

    async def _run_question_daytona(
        self,
        question: dict[str, Any],
        workspace: Path,
    ) -> dict[str, Any]:
        question_id = question["id"]

        logs_dir = workspace / "logs"
        (logs_dir / "agent").mkdir(parents=True, exist_ok=True)

        # Prepare local files
        image_filename = decode_and_save_image(question, workspace)
        prompt = self._build_prompt(question, workspace, image_filename)
        (workspace / "instruction.md").write_text(prompt, encoding="utf-8")

        sandbox = None
        try:
            # Create sandbox from python:3.11-slim (matches Docker base image)
            # Resources match Harbor defaults (1 CPU, 2GB RAM, 10GB disk)
            from daytona import Resources
            sandbox = await self._daytona.create(
                CreateSandboxFromImageParams(
                    image=Image.base("python:3.11-slim"),
                    resources=Resources(cpu=1, memory=2, disk=10),
                ),
                timeout=300,
            )

            # Install tools and Claude Code CLI (matches Harbor install script)
            cc_version = os.environ.get("CLAUDE_CODE_VERSION", "")
            cc_install = (
                f"curl -fsSL https://claude.ai/install.sh | bash -s -- {cc_version}"
                if cc_version
                else "curl -fsSL https://claude.ai/install.sh | bash"
            )
            _, _, rc = await self._daytona_exec(
                sandbox,
                "apt-get update && apt-get install -y curl procps && "
                "rm -rf /var/lib/apt/lists/* && "
                f"{cc_install} && "
                'echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.bashrc && '
                "export PATH=\"$HOME/.local/bin:$PATH\" && "
                "mkdir -p /app /logs/agent /logs/verifier",
                timeout=300,
            )
            if rc != 0:
                raise RuntimeError(f"Failed to install Claude Code CLI (exit {rc})")

            # Upload files
            await sandbox.fs.upload_file(
                str(workspace / "instruction.md"), "/app/instruction.md",
            )
            if image_filename and (workspace / image_filename).exists():
                await sandbox.fs.upload_file(
                    str(workspace / image_filename), f"/app/{image_filename}",
                )

            # Build env vars (Harbor-matching)
            env_vars = self._build_sandbox_env()

            # Setup Claude config dirs (matches Harbor)
            setup_cmd = (
                "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
                "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
                "$CLAUDE_CONFIG_DIR/todos && "
                "if [ -d ~/.claude/skills ]; then "
                "cp -r ~/.claude/skills $CLAUDE_CONFIG_DIR/skills 2>/dev/null || true; "
                "fi"
            )
            await self._daytona_exec(sandbox, setup_cmd, env=env_vars)

            # Build max-turns flag (matches Harbor)
            max_turns_flag = ""
            max_turns = os.environ.get("CLAUDE_CODE_MAX_TURNS")
            if max_turns:
                max_turns_flag = f"--max-turns {max_turns} "

            # Run Claude Code (matches Harbor CLI flags)
            claude_cmd = (
                'claude --verbose --output-format=stream-json '
                '--permission-mode bypassPermissions '
                f'{max_turns_flag}'
                '--print '
                '-- "# Task instructions are in /app/instruction.md\n'
                'Read that file for the complete task description." '
                '2>&1 </dev/null | tee /logs/agent/claude-code.txt'
            )
            stdout, stderr, returncode = await self._daytona_exec(
                sandbox, claude_cmd,
                env=env_vars,
                cwd="/app",
                timeout=int(self.config.timeout),
            )

            save_docker_outputs(workspace, stdout, stderr)

            # Download results
            try:
                await sandbox.fs.download_file(
                    "/logs/agent/response.txt",
                    str(logs_dir / "agent" / "response.txt"),
                )
            except Exception:
                pass
            try:
                await sandbox.fs.download_file(
                    "/logs/agent/claude-code.txt",
                    str(logs_dir / "agent" / "claude-code.txt"),
                )
            except Exception:
                pass

            metadata = {
                "question_id": question_id,
                "returncode": returncode,
                "timeout": False,
                "error": stderr if stderr else None,
                "runtime": "daytona",
                "sandbox_id": sandbox.id,
            }

        except asyncio.TimeoutError:
            metadata = {
                "question_id": question_id,
                "returncode": -1,
                "timeout": True,
                "error": f"Execution timed out after {self.config.timeout}s",
                "runtime": "daytona",
                "sandbox_id": sandbox.id if sandbox else None,
            }

        except Exception as e:
            metadata = {
                "question_id": question_id,
                "returncode": -1,
                "timeout": False,
                "error": str(e),
                "runtime": "daytona",
                "sandbox_id": sandbox.id if sandbox else None,
            }

        finally:
            if sandbox:
                try:
                    await self._daytona.delete(sandbox)
                except Exception:
                    pass

        parsed_answer = find_docker_response(logs_dir, workspace)

        raw_trajectory_file = logs_dir / "agent" / "claude-code.txt"
        trajectory = parse_claude_trajectory(raw_trajectory_file)
        if raw_trajectory_file.exists():
            raw_trajectory_file.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

        (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "question_id": question_id,
            "response": parsed_answer.get("raw", "") if parsed_answer else None,
            "parsed": parsed_answer,
            "trajectory": trajectory,
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
            elif self.environment == "daytona":
                return await self._run_question_daytona(question, workspace)
            return await self._run_question_local(question, workspace)

    async def run_all_questions(
        self,
        questions: list[dict[str, Any]],
        num_workers: int = 4,
    ) -> list[dict[str, Any]]:
        if self.environment == "docker":
            if not await docker_verify():
                raise RuntimeError("Docker daemon is not running")
        elif self.environment == "daytona":
            if not HAS_DAYTONA:
                raise ImportError(
                    "daytona package not installed. Run: pip install daytona"
                )
            if not os.environ.get("DAYTONA_API_KEY"):
                raise RuntimeError("DAYTONA_API_KEY environment variable is required")
            self._daytona = AsyncDaytona()

        try:
            semaphore = asyncio.Semaphore(num_workers)
            tasks = [self.run_question(q, semaphore) for q in questions]
            desc = f"Running questions ({self.environment})"
            return await tqdm_asyncio.gather(*tasks, desc=desc)
        finally:
            if self._daytona:
                try:
                    await self._daytona.close()
                except Exception:
                    pass
                self._daytona = None


# ── CLI entrypoint ──────────────────────────────────────────────────────

def main(args):
    config = ClaudeConfig(model=args.model, timeout=args.timeout)

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
    env_label = {"local": "claude-code", "docker": "claude-code-docker", "daytona": "claude-code-daytona"}
    prefix = env_label.get(args.environment, f"claude-code-{args.environment}")
    workspace_name = f"{prefix}_{args.model}_{timestamp}"
    workspace_root = (Path(args.output_dir) / workspace_name).resolve()
    print(f"Workspace: {workspace_root}")

    agent = ClaudeHLEAgent(config=config, workspace_root=workspace_root, environment=args.environment)
    results = asyncio.run(agent.run_all_questions(questions, num_workers=args.num_workers))

    output_file = workspace_root / "results.json"
    predictions = {}
    for result in results:
        predictions[result["question_id"]] = {
            "model": args.model,
            "response": result["response"],
            "parsed": result["parsed"],
            "trajectory": result["trajectory"],
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
        description="Run Claude Code CLI agent on HLE evaluation"
    )
    parser.add_argument("--dataset", type=str, default="cais/hle", help="HLE HuggingFace dataset name")
    parser.add_argument("--model", type=str, default="sonnet", help="Model name for claude CLI")
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
        choices=["local", "docker", "daytona"],
        default="local",
        help="Execution environment: local (default), docker, or daytona",
    )

    args = parser.parse_args()
    if args.sample_rate is not None and not (0.0 <= args.sample_rate <= 1.0):
        parser.error("--sample_rate must be between 0.0 and 1.0")
    main(args)
