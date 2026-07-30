# Korean Financial Input Guardrail — Auto-Tuning Framework

Official implementation for the paper:
**"A Regression-Aware Dynamic Guardrail Framework for Korean Financial PII Protection"**
*(Anonymized Repository for Review)*

A self-tuning guardrail pipeline for financial-domain LLM inputs. It runs a closed
loop: **red team (attack generation) → measurement (ASR/FPR) → blue team (patch
proposal) → regression check (McNemar test) → adopt / rollback / escalate.**

## Architecture

```
input text
   │
   ▼
Layer 1 (regex)   ──detected──▶ blocked
   │ pass
   ▼
Layer 2 (NER)     ──detected──▶ blocked
   │ pass
   ▼
Layer 3 (LLM, context + injection analysis) ──detected──▶ blocked
   │ pass
   ▼
allowed
```

Layer 3's decision criteria are stored as structured rules
(`guardrail/layer3_rules_v4.json`). When red-teaming finds a vulnerability, the
auto-tuning loop (1) proposes a local patch, (2) runs a McNemar significance
test against the accumulated regression corpus, (3) adopts the patch only if
it improves results with no regression (otherwise rolls back), and (4)
switches to a structural-refactor mode after two consecutive rollbacks of the
same failure type — all without human intervention.

## Structure

| Path | Description |
|---|---|
| `guardrail/` | 3-layer guardrail core (regex / NER / LLM) |
| `redteam/` | attack generation, ASR/FPR/latency measurement, multi-model comparison, ablation study |
| `blueteam/` | auto-tuning loop (v4, structured rules + escalation), false-positive fixes |
| `data/` | dataset generation/validation scripts |
| `experiments/` | baseline comparisons, held-out eval, caching/latency experiments |
| `cli.py` | unified entry point for the scripts above |
| `streamlit_demo.py` | live demo of detection, auto-tuning, and FP mitigation |
| `README_DEMO.md` | guide for running the demo |

## Usage

```bash
# run a single sentence through the 3-layer pipeline
python cli.py check "example sentence with a phone number"

# red-team evaluation (ASR/FPR/latency)
python cli.py redteam --dataset data/mixed_dataset_v3.json --output experiments/redteam_new.json

# initialize the v4 rule store (once)
python cli.py seed-rules

# run the v4 auto-tuning loop
python cli.py tune --dataset data/mixed_dataset_v3.json --threshold 0.05 --max-rounds 3 --auto-approve

# compare defender models
python cli.py multi-model --defender claude-sonnet --sample 50
python cli.py multi-model --defender llama-3 --sample 50

# compare against baseline tools
python cli.py baseline-presidio --korean-nlp
python cli.py baseline-llm-guard

# test Layer 3's own robustness to prompt injection
python cli.py judge-injection-test --n 100
```

Each subcommand is a thin wrapper around an independent script in
`redteam/`, `blueteam/`, or `experiments/` — logic is untouched, only the
interface is unified. Run `python cli.py <subcommand> -h` for a script's
actual options.

## Demo

```bash
pip install streamlit
streamlit run streamlit_demo.py
```

Three views: live detection, live auto-tuning (missed attack → patch
proposal → re-check), and false-positive mitigation (over-blocked input →
SAFE exception proposal → re-check). See `README_DEMO.md` for details.

## Reproducing from scratch

```bash
python cli.py seed-rules
python cli.py redteam --dataset data/mixed_dataset_v3.json
python cli.py tune --dataset data/mixed_dataset_v3.json --auto-approve
python cli.py multi-model --defender claude-sonnet
python cli.py multi-model --defender llama-3
python cli.py baseline-presidio
python cli.py baseline-llm-guard
python cli.py judge-injection-test
python experiments/caching_latency_test.py
```

## Setup

Requires an OpenAI API key (and optionally Anthropic/Hugging Face keys) set
as environment variables — see `requirements.txt` and the `os.getenv(...)`
calls in `redteam/`, `blueteam/`, and `guardrail/`. Do not commit a `.env`
file; it's already excluded via `.gitignore`.
