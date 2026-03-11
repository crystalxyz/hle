"""
Shared utilities for HLE evaluation agents (Claude Code, OpenHands, etc.).

Provides common functions for prompt building, image handling, answer parsing,
Docker lifecycle management, and dataset sampling.
"""
import re
import json
import shlex
import asyncio
import base64
import random
from pathlib import Path
from typing import Any, Callable
from collections import defaultdict


# ── Text utilities ──────────────────────────────────────────────────────

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(
        r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\[[\d;]*[a-zA-Z]|\[\?[\d;]*[a-zA-Z]'
    )
    return ansi_escape.sub('', text)


# ── Image handling ──────────────────────────────────────────────────────

MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
}


def get_image_extension(data_url: str) -> str:
    """Extract image file extension from a data URL header."""
    if not data_url.startswith("data:"):
        return ".png"
    header = data_url.split(",", 1)[0]
    for mime, ext in MIME_TO_EXT.items():
        if mime in header:
            return ext
    return ".png"


def decode_and_save_image(
    question: dict[str, Any],
    dest_dir: Path,
) -> str | None:
    """Decode a base64 data-URL image from an HLE question and save to dest_dir.

    Returns the image filename (e.g. "image.jpg") or None if no image.
    """
    if not question.get("image"):
        return None
    if not question["image"].startswith("data:"):
        return None
    extension = get_image_extension(question["image"])
    image_filename = f"image{extension}"
    _, encoded = question["image"].split(",", 1)
    (dest_dir / image_filename).write_bytes(base64.b64decode(encoded))
    return image_filename


# ── Prompt building ─────────────────────────────────────────────────────

def build_hle_prompt(
    question: dict[str, Any],
    response_path: str,
    image_path: str | None = None,
) -> str:
    """Build the standard HLE evaluation prompt.

    Args:
        question: HLE question dict (must have 'question' key).
        response_path: Absolute path where the agent should write its answer.
        image_path: Absolute path to the image file, or None if no image.
    """
    prompt = f"""# HLE Task

You are answering a challenging question that may require expert knowledge. You MUST write your answer to the file `{response_path}`.

"""
    if image_path:
        prompt += f"""Task: Read the image file `{image_path}` to answer the question.

"""
    prompt += f"""Question: {question['question']}

## Your Task

Write your answer to `{response_path}` in the exact format:

```
Explanation: <your reasoning for your answer>
Answer: <your final answer>
Confidence: <your confidence as a percentage, e.g., 85%>
```

"""
    return prompt


# ── Answer parsing ──────────────────────────────────────────────────────

def parse_answer_file(answer_file: Path) -> dict[str, str] | None:
    """Parse a response.txt file for Explanation, Answer, and Confidence.

    Returns dict with 'explanation', 'answer', 'confidence', 'raw' or None.
    """
    if not answer_file.exists():
        return None

    content = answer_file.read_text(encoding="utf-8")

    explanation_match = re.search(
        r"Explanation:\s*(.+?)(?=Answer:|Confidence:|$)",
        content, re.DOTALL | re.IGNORECASE,
    )
    answer_match = re.search(
        r"Answer:\s*(.+?)(?=Confidence:|$)",
        content, re.DOTALL | re.IGNORECASE,
    )
    confidence_match = re.search(
        r"Confidence:\s*(.+?)(?=$)",
        content, re.DOTALL | re.IGNORECASE,
    )

    if answer_match:
        return {
            "explanation": explanation_match.group(1).strip() if explanation_match else "",
            "answer": answer_match.group(1).strip(),
            "confidence": confidence_match.group(1).strip() if confidence_match else "",
            "raw": content,
        }

    return {
        "explanation": "",
        "answer": content.strip(),
        "confidence": "",
        "raw": content,
    }


def find_docker_response(
    logs_dir: Path,
    workspace: Path,
) -> dict[str, str] | None:
    """Search multiple locations for response.txt in Docker mode."""
    for candidate in [
        logs_dir / "agent" / "response.txt",
        workspace / "host_output" / "response.txt",
        workspace / "response.txt",
    ]:
        parsed = parse_answer_file(candidate)
        if parsed:
            return parsed
    return None


# ── Docker entrypoint templates ─────────────────────────────────────────

ENTRYPOINT_PREAMBLE = """\
#!/bin/bash
set -uo pipefail

# Ensure log directories exist (host volume mount may overwrite build-time dirs)
mkdir -p /logs/agent /logs/verifier

echo "================================================"
echo "HLE Runtime Setup"
echo "================================================"
echo "Files in /app/:"
ls -lah /app/
echo "  instruction.md: $(test -f /app/instruction.md && echo 'EXISTS' || echo 'MISSING')"
echo "  Image files: $(find /app -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.gif' -o -name '*.webp' 2>/dev/null | wc -l) found"
echo "================================================"
"""

ENTRYPOINT_POSTAMBLE = """\
EXIT_CODE=$?

echo "================================================"
echo "Agent finished with exit code: $EXIT_CODE"
echo "================================================"

# Copy results to mounted volumes so the host can access them
# /logs is mounted from the host logs directory
if [ -f /logs/agent/response.txt ]; then
    echo "Found response.txt at /logs/agent/response.txt"
else
    echo "WARNING: /logs/agent/response.txt not found"
    # Also check /app/response.txt as fallback
    if [ -f /app/response.txt ]; then
        mkdir -p /logs/agent
        cp /app/response.txt /logs/agent/response.txt
        echo "Copied /app/response.txt to /logs/agent/response.txt"
    fi
fi

# Logs are written directly to /logs/agent/ which is mounted from host
echo "Logs at /logs/agent/:"
ls -la /logs/agent/ 2>/dev/null || true

exit $EXIT_CODE
"""


def build_entrypoint(agent_command: str) -> str:
    """Build a complete entrypoint.sh from the shared preamble, an agent-specific
    command block, and the shared postamble."""
    return ENTRYPOINT_PREAMBLE + "\n" + agent_command + "\n" + ENTRYPOINT_POSTAMBLE


# ── Docker lifecycle helpers ────────────────────────────────────────────

async def docker_verify() -> bool:
    """Verify Docker daemon is running."""
    proc = await asyncio.create_subprocess_shell(
        "docker info",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if proc.returncode != 0:
        print("ERROR: Docker daemon is not running. Start Docker and try again.")
        return False
    return True


async def docker_build_task_image(
    question_id: str,
    build_ctx: Path,
    tag_prefix: str = "hle-task",
) -> str:
    """Build a Docker image for a single HLE task."""
    image_tag = f"{tag_prefix}-{question_id}"
    cmd = f"docker build -t {shlex.quote(image_tag)} {shlex.quote(str(build_ctx))}"

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()

    if proc.returncode != 0:
        build_log = stdout.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Docker build failed for task {question_id}:\n{build_log}"
        )
    return image_tag


async def docker_cleanup_task_image(image_tag: str) -> None:
    """Remove a per-task Docker image after use."""
    proc = await asyncio.create_subprocess_shell(
        f"docker rmi {shlex.quote(image_tag)} 2>/dev/null || true",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


# ── Docker build context ────────────────────────────────────────────────

def create_docker_build_context(
    question: dict[str, Any],
    workspace: Path,
    dockerfile: str,
    entrypoint: str,
    build_prompt: Callable[[dict[str, Any], str | None], str],
) -> tuple[Path, str | None]:
    """Create a Docker build context directory with Dockerfile, entrypoint.sh,
    and workspace/ containing instruction.md and any image files.

    Args:
        question: HLE question dict.
        workspace: Per-question workspace directory on the host.
        dockerfile: Dockerfile content string.
        entrypoint: entrypoint.sh content string.
        build_prompt: Callable(question, image_filename) -> prompt string.

    Returns:
        Tuple of (build_context_dir, image_filename or None).
    """
    build_ctx = workspace / "docker_build"
    build_ctx.mkdir(parents=True, exist_ok=True)

    (build_ctx / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (build_ctx / "entrypoint.sh").write_text(entrypoint, encoding="utf-8")

    ws_dir = build_ctx / "workspace"
    ws_dir.mkdir(exist_ok=True)

    image_filename = decode_and_save_image(question, ws_dir)

    prompt_content = build_prompt(question, image_filename)
    (ws_dir / "instruction.md").write_text(prompt_content, encoding="utf-8")

    return build_ctx, image_filename


# ── Docker output saving ───────────────────────────────────────────────

def save_docker_outputs(
    workspace: Path,
    stdout_text: str,
    stderr_text: str,
) -> None:
    """Save raw and cleaned stdout/stderr from a Docker run."""
    if stdout_text:
        (workspace / "stdout_raw.txt").write_text(stdout_text, encoding="utf-8")
        (workspace / "stdout_clean.txt").write_text(
            strip_ansi_codes(stdout_text), encoding="utf-8"
        )
    if stderr_text:
        (workspace / "stderr_raw.txt").write_text(stderr_text, encoding="utf-8")


# ── Dataset sampling ────────────────────────────────────────────────────

def stratified_sample(
    questions: list[dict[str, Any]],
    sample_rate: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample from each HLE category at the given rate."""
    random.seed(seed)

    category_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        category = question.get("category", "unknown")
        category_questions[category].append(question)

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
