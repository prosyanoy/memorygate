"""Markdown rendering of gate reports and single runs."""

from __future__ import annotations

from .models import GateReport, RunResult

MARKER = "<!-- memorygate-report -->"
ICONS = {"regression": "❌ regression", "improved": "✅ improved", "ok": "➖ no change"}


def _label(name: str, k: int) -> str:
    return f"hit@{k}" if name == "hit_at_k" else name


def render_gate(report: GateReport) -> str:
    verdict = "✅ **PASSED**" if report.passed else "❌ **FAILED** - statistically meaningful quality drop"
    lines = [
        MARKER,
        "## MemoryGate report",
        "",
        f"{verdict}",
        "",
        f"Baseline `{report.baseline_adapter}` vs candidate `{report.candidate_adapter}` "
        f"on {report.n_customers} customers.",
        "",
        f"| Metric | Better when | Baseline | Candidate | Δ | {report.confidence:.0%} CI of Δ | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for m in report.metrics:
        lines.append(
            f"| `{_label(m.name, report.k)}` | {'higher' if m.higher_is_better else 'lower'} "
            f"| {m.baseline:.3f} | {m.candidate:.3f} | {m.delta:+.3f} "
            f"| [{m.delta_ci_low:+.3f}, {m.delta_ci_high:+.3f}] | {ICONS[m.status]} |"
        )
    lines += [
        "",
        f"<sub>Paired bootstrap over customers ({report.n_bootstrap} resamples). A metric fails when its "
        f"one-sided {report.confidence:.0%} bound on the worsening is above 0 and the drop is at least "
        f"{report.min_effect:g}.</sub>",
        "",
    ]
    return "\n".join(lines)


def render_run(run: RunResult) -> str:
    lines = [f"## MemoryGate run: `{run.adapter}`", "", "| Metric | Value |", "|---|---:|"]
    for k, v in run.metrics.items():
        lines.append(f"| `{_label(k, run.k)}` | {v:.3f} |")
    return "\n".join(lines) + "\n"
