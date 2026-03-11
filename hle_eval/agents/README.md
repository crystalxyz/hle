# HLE Eval Agents

Agent implementations for running [Humanity's Last Exam (HLE)](https://huggingface.co/datasets/cais/hle) evaluations. Used for parity experiments against Harbor HLE adapter.

## Agents

| Agent | File | Description |
|-------|------|-------------|
| Claude Code | `claude_agent_hle.py` | Claude Code CLI agent. Supports local, Docker, and Daytona execution. |
| OpenHands | `openhands_agent_hle.py` | OpenHands (CodeAct) agent. Supports local and Docker execution. |
| Codex | `codex_agent_hle.py` | OpenAI Codex CLI agent. |

## Shared Modules

| File | Description |
|------|-------------|
| `hle_common.py` | Shared utilities: prompt building, image handling, answer parsing, Docker lifecycle, dataset sampling. |
| `claude_config.py` | Claude Code configuration (API key, model, base URL, timeout). |
| `openhands_config.py` | OpenHands configuration. |
| `judge_agent_results.py` | LLM judge for evaluating agent responses (default: gpt-5). |

## Reproduction step

### Claude Code

```bash
# Local execution
python agents/claude_agent_hle.py --model claude-haiku-4-5 --max_samples 5

# Docker (per-task containers)
python agents/claude_agent_hle.py --model claude-haiku-4-5 -e docker --max_samples 5

# Daytona (cloud sandboxes)
DAYTONA_API_KEY=... python claude_agent_hle.py --model claude-haiku-4-5 -e daytona --sample_rate 0.1 --num_workers 10 --max_samples 10
```

```

### Judging Results

```bash
# Judge a workspace
python agents/judge_agent_results.py --workspace ../jobs/claude-code_claude-haiku-4-5_20260309_123456

# Judge a predictions file
python agents/judge_agent_results.py --predictions ../jobs/results.json --judge gpt-5
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `ANTHROPIC_BASE_URL` | No | Custom API base URL (e.g., proxy) |
| `DAYTONA_API_KEY` | Daytona only | Daytona API key |
| `OPENAI_API_KEY` | Judge only | OpenAI API key for LLM judge |
| `MAX_THINKING_TOKENS` | No | Limit thinking tokens for Claude |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | No | Limit output tokens |
| `CLAUDE_CODE_MAX_TURNS` | No | Limit agent turns |

## Common Options

```
--model MODEL          Model name (default: sonnet)
--timeout TIMEOUT      Timeout per question in seconds (default: 1200)
--num_workers N        Concurrent workers (default: 4)
--task_ids ID [ID ...] Run specific task IDs only
--sample_rate RATE     Stratified sampling rate (0.0-1.0)
--max_samples N        Limit total samples
--output_dir DIR       Output directory (default: ../jobs)
```

## Output Structure

Each run creates a workspace under `../jobs/`:

```
jobs/claude-code-daytona_claude-haiku-4-5_20260309_211410/
├── results.json                          # Aggregated results
├── run_<question_id>/
│   ├── metadata.json                     # Exit code, timeout, runtime info
│   ├── instruction.md                    # Prompt sent to agent
│   ├── logs/agent/
│   │   ├── response.txt                  # Agent's answer
│   │   └── claude-code.txt               # Raw trajectory
│   ├── stdout_raw.txt                    # Container stdout
│   └── stdout_clean.txt                  # ANSI-stripped stdout
└── judged_results.json                   # Judge output (after judging)
```
