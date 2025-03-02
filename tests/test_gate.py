import numpy as np
import pytest

from memorygate.gate import GateError, compare
from memorygate.models import CustomerStats, GeneratorConfig, RunResult
from memorygate.report import render_gate


def make_run(name, rng_seed, p_hit, n=80, leak=0.0, fp=None, fingerprint="fp"):
    """Synthetic per-customer counts: 10 queries/customer, `p_hit` chance of a hit each."""
    rng = np.random.default_rng(rng_seed)
    per = {}
    for i in range(n):
        hits = float(rng.binomial(10, p_hit))
        per[f"c{i:03d}"] = CustomerStats(
            tp=hits, fp=0 if fp is None else float(rng.binomial(10, fp)), fn=10 - hits,
            stale_hits=0, stale_total=2, leaked=float(rng.binomial(30, leak)), retrieved=30,
            hits=hits, rr_sum=hits, queries=10)
    return RunResult(adapter=name, k=3, dataset_config=GeneratorConfig(), dataset_fingerprint=fingerprint,
                     metrics={}, per_customer=per)


def by_name(report):
    return {m.name: m for m in report.metrics}


def test_identical_runs_pass_with_zero_delta():
    base = make_run("a", 1, 0.9)
    rep = compare(base, base)
    assert rep.passed
    assert all(m.delta == 0 and m.status == "ok" for m in rep.metrics)


def test_large_drop_fails():
    rep = compare(make_run("a", 1, 0.9), make_run("b", 2, 0.6))
    assert not rep.passed
    assert by_name(rep)["hit_at_k"].status == "regression"
    assert by_name(rep)["fact_recall"].delta < -0.2


def test_improvement_is_reported_not_blocked():
    rep = compare(make_run("a", 1, 0.6), make_run("b", 2, 0.9))
    assert rep.passed and by_name(rep)["hit_at_k"].status == "improved"


def test_leakage_increase_fails_lower_is_better_direction():
    rep = compare(make_run("a", 1, 0.9, leak=0.0), make_run("b", 1, 0.9, leak=0.2))
    assert not rep.passed
    m = by_name(rep)["leakage_rate"]
    assert m.status == "regression" and m.delta > 0


def test_noise_level_difference_is_not_significant():
    # same underlying quality, different random draws
    rep = compare(make_run("a", 1, 0.8), make_run("b", 2, 0.8))
    assert by_name(rep)["hit_at_k"].status != "regression"


def test_min_effect_blocks_tiny_but_significant_drop():
    base = make_run("a", 5, 0.90, n=400)
    cand = make_run("a", 5, 0.90, n=400)
    # shift every customer by one fewer hit on 1 in 10 customers: ~1% drop, paired => tight CI
    for i, st in enumerate(cand.per_customer.values()):
        if i % 10 == 0 and st.hits > 0:
            st.hits -= 1; st.rr_sum -= 1
    strict = compare(base, cand, min_effect=0.0)
    lenient = compare(base, cand, min_effect=0.05)
    assert by_name(strict)["hit_at_k"].status == "regression"
    assert by_name(lenient)["hit_at_k"].status != "regression"


def test_mismatches_raise():
    with pytest.raises(GateError, match="fingerprint"):
        compare(make_run("a", 1, 0.9), make_run("b", 1, 0.9, fingerprint="other"))
    short = make_run("b", 1, 0.9, n=10)
    with pytest.raises(GateError, match="different customers"):
        compare(make_run("a", 1, 0.9), short)


def test_bootstrap_is_reproducible():
    a, b = make_run("a", 1, 0.9), make_run("b", 2, 0.85)
    assert compare(a, b, seed=3).model_dump() == compare(a, b, seed=3).model_dump()


def test_markdown_report_contents():
    md = render_gate(compare(make_run("a", 1, 0.9), make_run("b", 2, 0.6)))
    assert "FAILED" in md and "hit@3" in md and "regression" in md and md.startswith("<!-- memorygate-report -->")
