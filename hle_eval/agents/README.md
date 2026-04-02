# HLE Eval Agents

Agent implementations for running [Humanity's Last Exam (HLE)](https://huggingface.co/datasets/cais/hle) evaluations. Used for parity experiments against Harbor HLE adapter.

## Agents

| Agent | File | Description |
|-------|------|-------------|
| Claude Code | `claude_agent_hle.py` | Claude Code CLI agent. Supports local, Docker, and Daytona execution. |
| Codex | `codex_agent_hle.py` | OpenAI Codex CLI agent. Supports local and Docker execution. |

## Shared Modules

| File | Description |
|------|-------------|
| `hle_common.py` | Shared utilities: prompt building, image handling, answer parsing, Docker lifecycle, dataset sampling. |
| `claude_config.py` | Claude Code configuration (API key, model, base URL, timeout). |
| `judge_agent_results.py` | LLM judge for evaluating agent responses (default: gpt-5). |

## Parity Results

The parity subset (249 tasks) is drawn by randomly selecting 10% of each category with random seed 42. All experiments use gpt-5 as the LLM judge.

| Agent              | Model            | Metric                | Trials | Dataset Size          | Original Benchmark | Harbor Adapter |
| ------------------ | ---------------- | --------------------- | ------ | --------------------- | ------------------ | -------------- |
| claude-code@2.1.76 | claude-haiku-4-5 | Accuracy (%)          | 3      | 249 (10% of full set) | 10.71% ± 0.94%     | 10.98% ± 0.36% |
| claude-code@2.1.76 | claude-haiku-4-5 | Calibration error (%) | 3      | 249 (10% of full set) | 55.22% ± 0.59%     | 52.69% ± 0.67% |

- Uncertainties are sampling standard error of the mean.
- Calibration error uses ECE with beta=10 (binned L2-norm).
- Full results: see `adapters/hle/parity_experiment.json` in the Harbor adapter directory.

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

### Judging Results

```bash
# Judge a workspace
python agents/judge_agent_results.py --workspace ../jobs/claude-code_claude-haiku-4-5_xxx

# Judge a predictions file
python agents/judge_agent_results.py --predictions ../jobs/results.json --judge gpt-5
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude only | Anthropic API key |
| `ANTHROPIC_BASE_URL` | No | Custom API base URL (e.g., proxy) |
| `OPENAI_API_KEY` | Codex/Judge | OpenAI API key |
| `DAYTONA_API_KEY` | Daytona only | Daytona API key |

## Common Options

```
--model MODEL          Model name
--timeout TIMEOUT      Timeout per question in seconds (default: 1200)
--num_workers N        Concurrent workers (default: 4)
--task_ids ID [ID ...] Run specific task IDs only
--task_ids_file FILE   JSON file with task IDs to run
--sample_rate RATE     Stratified sampling rate (0.0-1.0)
--sample_seed SEED     Random seed for sampling (default: 42)
--max_samples N        Limit total samples
--output_dir DIR       Output directory (default: ../jobs)
-e, --environment      Execution environment (local, docker, daytona)
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
