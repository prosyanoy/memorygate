"""Regression gate: paired bootstrap over customers against a saved baseline.

Both runs are scored on the same dataset, so each bootstrap resample draws the
same customers for baseline and candidate. For every metric we compute the
*worsening* (sign-adjusted so positive = worse) and fail only if its one-sided
lower confidence bound is above zero AND the point estimate exceeds
`min_effect`. Noise-level wobble therefore never blocks a PR, while a real,
repeatable drop does.
"""

from __future__ import annotations

import numpy as np

from .metrics import HIGHER_IS_BETTER, metric_values, stats_matrix
from .models import GateReport, MetricComparison, RunResult


class GateError(Exception):
    """The two runs cannot be compared (different dataset, k, customers)."""


def compare(
    baseline: RunResult,
    candidate: RunResult,
    *,
    confidence: float = 0.95,
    min_effect: float = 0.01,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> GateReport:
    if baseline.dataset_fingerprint != candidate.dataset_fingerprint:
        raise GateError(
            "dataset fingerprint mismatch "
            f"({baseline.dataset_fingerprint} vs {candidate.dataset_fingerprint}); "
            "the generator or its config changed - regenerate the baseline."
        )
    if baseline.k != candidate.k:
        raise GateError(f"k mismatch ({baseline.k} vs {candidate.k})")
    ids = sorted(baseline.per_customer)
    if ids != sorted(candidate.per_customer):
        raise GateError("baseline and candidate cover different customers")

    base_m = stats_matrix(baseline.per_customer, ids)
    cand_m = stats_matrix(candidate.per_customer, ids)
    n = len(ids)
    rng = np.random.default_rng(seed)
    # multinomial weights == resampling n customers with replacement
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_bootstrap).astype(float)
    base_boot = metric_values(weights @ base_m)
    cand_boot = metric_values(weights @ cand_m)
    base_pt = metric_values(base_m.sum(axis=0))
    cand_pt = metric_values(cand_m.sum(axis=0))

    alpha = 1.0 - confidence
    rows: list[MetricComparison] = []
    for name, higher in HIGHER_IS_BETTER.items():
        sign = 1.0 if higher else -1.0
        delta_boot = cand_boot[name] - base_boot[name]
        delta = float(cand_pt[name] - base_pt[name])
        worsening = -sign * delta_boot
        bound = float(np.quantile(worsening, alpha))
        improve_bound = float(np.quantile(worsening, 1.0 - alpha))
        point_worse = -sign * delta
        if bound > 0 and point_worse >= min_effect:
            status = "regression"
        elif improve_bound < 0:
            status = "improved"
        else:
            status = "ok"
        rows.append(
            MetricComparison(
                name=name,
                higher_is_better=higher,
                baseline=float(base_pt[name]),
                candidate=float(cand_pt[name]),
                delta=delta,
                delta_ci_low=float(np.quantile(delta_boot, alpha / 2)),
                delta_ci_high=float(np.quantile(delta_boot, 1 - alpha / 2)),
                worsening_bound=bound,
                status=status,
            )
        )
    return GateReport(
        passed=all(r.status != "regression" for r in rows),
        baseline_adapter=baseline.adapter,
        candidate_adapter=candidate.adapter,
        k=baseline.k,
        n_customers=n,
        confidence=confidence,
        min_effect=min_effect,
        n_bootstrap=n_bootstrap,
        metrics=rows,
    )
