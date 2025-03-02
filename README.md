# MemoryGate

A benchmark and regression-gate framework that scores an AI memory system on **accuracy, staleness and cross-customer leakage** and **blocks a pull request if quality drops**.

> Designed an eval harness with regression gates (bootstrap CIs) for LLM memory quality: precision/recall, staleness and cross-customer leakage.

### What a failed gate looks like

Real output of `memorygate gate --adapter worse` against the committed baseline (exit code 1). In CI the same Markdown is posted as the PR comment and job summary. This is the CLI's output pasted as text, not a screenshot of a GitHub PR check; the workflow has not been run on GitHub yet.

❌ **FAILED** - statistically meaningful quality drop

Baseline `better` vs candidate `worse` on 60 customers.

| Metric | Better when | Baseline | Candidate | Δ | 95% CI of Δ | Status |
|---|---|---:|---:|---:|---:|---|
| `fact_precision` | higher | 0.943 | 0.397 | -0.546 | [-0.583, -0.508] | ❌ regression |
| `fact_recall` | higher | 0.895 | 0.559 | -0.336 | [-0.381, -0.290] | ❌ regression |
| `fact_f1` | higher | 0.918 | 0.464 | -0.454 | [-0.491, -0.416] | ❌ regression |
| `stale_fact_rate` | lower | 0.062 | 0.734 | +0.672 | [+0.618, +0.725] | ❌ regression |
| `leakage_rate` | lower | 0.000 | 0.487 | +0.487 | [+0.449, +0.524] | ❌ regression |
| `hit@3` | higher | 0.895 | 0.522 | -0.373 | [-0.421, -0.328] | ❌ regression |
| `mrr` | higher | 0.895 | 0.380 | -0.515 | [-0.561, -0.470] | ❌ regression |

<sub>Paired bootstrap over customers (2000 resamples). A metric fails when its one-sided 95% bound on the worsening is above 0 and the drop is at least 0.01.</sub>

### Demo walkthrough (reproduce the "degrade, fail, fix" story)

1. On a branch, make the memory system worse. Locally, the quickest way is the built-in flawed adapter; with a real system it would be a bad prompt change, e.g. removing the "ignore facts about other people / keep only the corrected value" lines from the extraction prompt in `adapters/ollama.py`.
2. Run `memorygate gate --adapter worse` (or `--adapter ollama` with Ollama running). It prints the table above and exits 1; in a PR the `memorygate` check turns red and the comment appears.
3. Revert the change, re-run `memorygate gate --adapter better`: all metrics are `no change`, exit 0, check goes green.

_Screenshot of the red PR check and a screen recording of this flow: to be added once the workflow has run on a real PR._

## Features

- **Seeded synthetic dataset** – templated multi-session support conversations with ground-truth facts, *updates* (old value becomes stale), planted *contradictions* (wrong value, then a correction), third-party statements ("my sister's city is…"), outdated assistant records, and *distractor customers* with overlapping values.
- **Metrics suite** – fact precision / recall / F1, stale-fact rate, wrong-customer leakage rate, retrieval hit@k and MRR.
- **Baseline store + regression gate** – the candidate is compared to a saved baseline with a paired bootstrap over customers; exits non-zero only on a statistically meaningful drop.
- **GitHub Actions workflow** – runs the gate on every PR, posts (and updates) a Markdown metric diff as a PR comment, writes the job summary, uploads JSON/Markdown reports.
- **Pluggable adapters** – a three-method `MemorySystem` interface. Built-ins: `better` (rule-based extractor, tenant-isolated, last-write-wins), `worse` (deliberately flawed), `ollama` (local LLM extractor). Your own system plugs in as `package.module:factory`.

## Architecture

```
generator.py  seeded Dataset (customers, conversations, queries, ground truth)
     │
runner.py     ingests conversations chronologically into ONE shared MemorySystem,
     │        then calls memories(customer) and retrieve(customer, question, k)
metrics.py    per-customer raw counts  ──►  ratios (precision, recall, F1, stale, leakage, hit@k, MRR)
     │
RunResult ──► baselines/baseline.json   (counts per customer + dataset config + fingerprint)
     │
gate.py       paired bootstrap over customers: candidate vs baseline ──► GateReport
report.py     Markdown / JSON        cli.py  exit 0 pass · 1 regression · 2 error
adapters/     base (interface) · scoped (isolated store) · rules · naive · ollama · registry
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional; every setting has a default, no secrets needed
```

## Run

```bash
memorygate adapters                       # list built-in adapters
memorygate run --adapter better           # score an adapter (also: worse)
memorygate baseline --adapter better      # (re)record baselines/baseline.json
memorygate gate --adapter better          # passes, exit 0
memorygate gate --adapter worse           # fails, exit 1, prints the Markdown diff
memorygate gate --adapter worse --report-md out/report.md --report-json out/report.json
```

Reference numbers on the shipped dataset (seed 7, 60 customers, k=3):

| | precision | recall | stale rate | leakage | hit@3 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| `better` | 0.943 | 0.895 | 0.062 | 0.000 | 0.895 | 0.895 |
| `worse` | 0.397 | 0.559 | 0.734 | 0.487 | 0.522 | 0.380 |

**Local LLM:** start Ollama, `ollama pull llama3.2`, then `memorygate run --adapter ollama` (`OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT` are read from the environment). If the server is unreachable the CLI exits with code 2 and a clear message.

**Your own system:** subclass `memorygate.adapters.MemorySystem` (`ingest`, `memories`, `retrieve`; return `MemoryItem`s whose `conversation_id` is the conversation they came from), expose a zero-arg factory, and run `memorygate gate --adapter mypkg.memory:build`. In CI set the repository variable `MEMORYGATE_ADAPTER`.

Exit codes: `0` pass, `1` regression, `2` error (missing baseline, dataset/generator drift, adapter failure).

## Test

```bash
pytest -q
```

Covers the generator (determinism, planted truth), metric formulas, adapters (incl. Ollama through a fake transport), the bootstrap gate (identical runs, real drops, noise, improvements, lower-is-better direction, `min_effect`), the CLI and the shipped baseline.

## Key design decisions

- **Leakage is judged by provenance, not by what the adapter claims.** Every returned `MemoryItem` carries its source `conversation_id`; the dataset knows which customer owns each conversation. An item from another customer's conversation is a leak even if its value happens to coincide. Leakage rate = leaked items / items returned by retrieval (top-k).
- **Stale rate** = stale `(attribute, value)` pairs still present in the system's active memories / stale pairs planted (updates and contradictions). **Precision** counts stale, third-party, unstructured and foreign memories as false positives.
- **Bootstrap over customers, paired.** Customers are the independent unit, so per-customer counts are stored in the baseline and resampled (multinomial weights, identical for baseline and candidate). No need to re-run the baseline system in CI.
- **Decision rule.** A metric fails when the one-sided 95 % bound on its *worsening* is above 0 **and** the point drop is ≥ `--min-effect` (default 0.01). The reported CI is the two-sided interval of Δ. Improvements are shown, never blocking. Any single regressed metric blocks the PR.
- **Gate regenerates the dataset from the baseline's stored config** and checks a fingerprint, so a changed generator or config is an explicit error (exit 2) rather than a silent apples-to-oranges comparison; re-record the baseline deliberately.
- **Baseline is the committed `better` adapter.** In a real project, record the baseline from your system on `main` and commit it; the demo adapters live in the package so the gate can be shown failing and passing without extra setup.
- **Deterministic, offline by default.** The stub (rule-based) path needs no network or secrets; Ollama is opt-in. The workflow's comment step is best-effort (fork PRs have read-only tokens) while the gate step still fails the check.
- **Intentional imperfection.** The `better` adapter misses a few "hard" phrasings the generator emits (`hard_prob`), so its scores are realistic (not 1.0) and the bootstrap intervals are meaningful.
