import argparse
import csv
import json
import math
from pathlib import Path

try:
    from artifact_paths import (
        janus_scores_path,
        panel_path as round_panel_path,
        panel_with_janus_path as round_panel_with_janus_path,
        panel_with_janus_summary_path as round_panel_with_janus_summary_path,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        janus_scores_path,
        panel_path as round_panel_path,
        panel_with_janus_path as round_panel_with_janus_path,
        panel_with_janus_summary_path as round_panel_with_janus_summary_path,
    )


def safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def normal_pdf(x: float, mu: float, var: float) -> float:
    var = max(var, 1e-9)
    z = (x - mu) / math.sqrt(var)
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi * var)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fit_two_gaussian_mixture(values, max_iter: int = 200, tol: float = 1e-7):
    """Simple 2-component EM fit for 1D values.

    Returns a dict with component params and helper metadata, or None if not fit.
    """
    xs = [float(v) for v in values if v is not None and math.isfinite(v)]
    n = len(xs)
    if n < 8:
        return None

    sxs = sorted(xs)
    mu0 = sxs[max(0, int(0.25 * (n - 1)))]
    mu1 = sxs[max(0, int(0.75 * (n - 1)))]
    mean_all = sum(xs) / n
    var_all = max(1e-6, sum((x - mean_all) ** 2 for x in xs) / n)

    pi0 = 0.5
    pi1 = 0.5
    var0 = var_all
    var1 = var_all
    last_ll = None

    for _ in range(max_iter):
        # E-step
        r0 = []
        ll = 0.0
        for x in xs:
            p0 = pi0 * normal_pdf(x, mu0, var0)
            p1 = pi1 * normal_pdf(x, mu1, var1)
            denom = p0 + p1
            if denom <= 0.0:
                gamma0 = 0.5
                ll += -1e6
            else:
                gamma0 = p0 / denom
                ll += math.log(max(denom, 1e-300))
            r0.append(gamma0)

        # M-step
        n0 = sum(r0)
        n1 = n - n0
        if n0 < 1e-6 or n1 < 1e-6:
            return None

        pi0 = max(1e-6, min(1.0 - 1e-6, n0 / n))
        pi1 = 1.0 - pi0
        mu0 = sum(g * x for g, x in zip(r0, xs)) / n0
        mu1 = sum((1.0 - g) * x for g, x in zip(r0, xs)) / n1
        var0 = sum(g * (x - mu0) ** 2 for g, x in zip(r0, xs)) / n0
        var1 = sum((1.0 - g) * (x - mu1) ** 2 for g, x in zip(r0, xs)) / n1
        var0 = max(var0, 1e-6)
        var1 = max(var1, 1e-6)

        if last_ll is not None and abs(ll - last_ll) < tol:
            break
        last_ll = ll

    # lower-mean component is interpreted as destabilizing mode
    if mu0 <= mu1:
        low = {"pi": pi0, "mu": mu0, "var": var0}
        high = {"pi": pi1, "mu": mu1, "var": var1}
    else:
        low = {"pi": pi1, "mu": mu1, "var": var1}
        high = {"pi": pi0, "mu": mu0, "var": var0}
    return {"low": low, "high": high}


def posterior_low_component(x: float, mix) -> float:
    if mix is None:
        return 0.5
    low = mix["low"]
    high = mix["high"]
    pl = low["pi"] * normal_pdf(x, low["mu"], low["var"])
    ph = high["pi"] * normal_pdf(x, high["mu"], high["var"])
    denom = pl + ph
    if denom <= 0.0:
        return 0.5
    return pl / denom


def posterior_destabilizing_component(x: float, mix, janus_positive_stabilizing: bool = True) -> float:
    """Posterior probability that x belongs to destabilizing mixture component."""
    p_low = posterior_low_component(x, mix)
    if janus_positive_stabilizing:
        # Janus convention: larger DDG is more stabilizing -> low mean mode is destabilizing.
        return p_low
    # Classic convention: smaller DDG is more stabilizing -> high mean mode is destabilizing.
    return 1.0 - p_low


def bayes_posterior_fdr_outliers(ddg_by_id, mix, q: float, janus_positive_stabilizing: bool = True):
    """Bayesian FDR control from destabilizing posteriors.

    Sort by P(destabilizing|x) descending and keep the largest prefix whose
    expected FDR = mean(1 - P(destabilizing|x)) is <= q.
    """
    if mix is None:
        return set()
    q = max(1e-6, min(0.5, float(q)))
    ranked = []
    for vid, x in ddg_by_id.items():
        p_destab = max(0.0, min(1.0, posterior_destabilizing_component(x, mix, janus_positive_stabilizing)))
        ranked.append((vid, p_destab))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return set()

    best_k = 0
    cum_false = 0.0
    for i, (_vid, p_destab) in enumerate(ranked, start=1):
        cum_false += (1.0 - p_destab)
        fdr_hat = cum_false / float(i)
        if fdr_hat <= q:
            best_k = i
    if best_k <= 0:
        return set()
    return {vid for vid, _p in ranked[:best_k]}


def robust_location_scale(values):
    xs = [float(v) for v in values if v is not None and math.isfinite(v)]
    n = len(xs)
    if n < 8:
        return None, None
    s = sorted(xs)
    mid = n // 2
    if n % 2 == 1:
        med = s[mid]
    else:
        med = 0.5 * (s[mid - 1] + s[mid])
    abs_dev = sorted(abs(x - med) for x in xs)
    if n % 2 == 1:
        mad = abs_dev[mid]
    else:
        mad = 0.5 * (abs_dev[mid - 1] + abs_dev[mid])
    # 1.4826*MAD is robust sigma estimator for normal-like cores.
    sigma = 1.4826 * mad
    if sigma < 1e-9:
        return med, None
    return med, sigma


def bh_fdr_flags(pairs, q: float = 0.10):
    """Benjamini-Hochberg FDR control.

    pairs: list[(id, pvalue)] ; returns set(ids) rejected at FDR q.
    """
    clean = [(k, p) for k, p in pairs if p is not None and math.isfinite(p)]
    m = len(clean)
    if m == 0:
        return set()
    clean.sort(key=lambda kv: kv[1])
    cutoff = None
    for i, (_, p) in enumerate(clean, start=1):
        if p <= q * i / m:
            cutoff = p
    if cutoff is None:
        return set()
    return {k for k, p in clean if p <= cutoff}


def detect_id_col(fieldnames):
    for c in ("ID", "id", "variant_id", "mutation_id", "Mutation"):
        if c in fieldnames:
            return c
    for c in fieldnames:
        if "id" in c.lower():
            return c
    return None


def detect_ddg_col(rows, fieldnames):
    preferred = [
        "ddg", "DDG", "delta_ddg", "ΔΔG", "pred_ddg", "prediction", "score"
    ]
    lower_map = {c.lower(): c for c in fieldnames}
    for p in preferred:
        c = lower_map.get(p.lower())
        if c:
            return c
    best = None
    best_n = -1
    for c in fieldnames:
        if "id" in c.lower() or "seq" in c.lower() or c.lower() in ("mts", "mutation"):
            continue
        n = 0
        for r in rows[:200]:
            if safe_float(r.get(c)) is not None:
                n += 1
        if n > best_n:
            best_n = n
            best = c
    return best


def normalize_row_dict(row: dict) -> dict:
    out = {}
    for k, v in (row or {}).items():
        kk = str(k or "").strip()
        if not kk:
            continue
        vv = v.strip() if isinstance(v, str) else v
        out[kk] = vv
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Join JanusDDG scores onto SWARM panel and rank final triage set.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--panel", default=None, help="Default: OUTDIR/swarm/swarm_panel_rK.tsv")
    ap.add_argument("--janus", default=None, help="Default: OUTDIR/swarm/janus_scores_rK.csv")
    ap.add_argument("--out", default=None, help="Default: OUTDIR/swarm/swarm_panel_with_janus_rK.tsv")
    ap.add_argument("--summary", default=None, help="Default: OUTDIR/swarm/swarm_panel_with_janus_summary_rK.json")
    ap.add_argument(
        "--stability-gate",
        choices=["bayes"],
        default="bayes",
        help="Bayesian posterior-BFDR gate (no fixed ddG cutoff).",
    )
    ap.add_argument(
        "--stability-fdr",
        type=float,
        default=0.10,
        help="FDR level for adaptive Janus outlier detection under the bayes gate.",
    )
    ap.add_argument(
        "--drop-janus-outliers",
        action="store_true",
        default=True,
        help="When using bayes gate, remove rows classified as destabilizing outliers by FDR.",
    )
    ap.add_argument(
        "--keep-janus-outliers",
        dest="drop_janus_outliers",
        action="store_false",
        help="Keep Janus outliers in output and only annotate flags.",
    )
    ap.add_argument(
        "--janus-positive-stabilizing",
        action="store_true",
        default=True,
        help="Interpret larger DDG as more stabilizing (JanusDDG convention).",
    )
    ap.add_argument(
        "--negative-stabilizing",
        dest="janus_positive_stabilizing",
        action="store_false",
        help="Interpret smaller DDG as more stabilizing (classic destabilizing-ddG convention).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    round_id = int(args.round)
    panel_path = Path(args.panel) if args.panel else round_panel_path(outdir=outdir, round_id=round_id)
    janus_path = Path(args.janus) if args.janus else janus_scores_path(outdir=outdir, round_id=round_id)
    out_path = Path(args.out) if args.out else round_panel_with_janus_path(outdir=outdir, round_id=round_id)
    summary_path = Path(args.summary) if args.summary else round_panel_with_janus_summary_path(outdir=outdir, round_id=round_id)

    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}")
    if not janus_path.exists():
        raise SystemExit(f"Janus output not found: {janus_path}")

    with janus_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = [normalize_row_dict(r) for r in reader]
        fields = [str(f).strip() for f in (reader.fieldnames or [])]

    if not fields:
        raise SystemExit("Janus CSV has no header.")
    id_col = detect_id_col(fields)
    ddg_col = detect_ddg_col(rows, fields)
    if not id_col or not ddg_col:
        raise SystemExit(f"Could not detect ID/DDG columns in Janus CSV. fields={fields}")

    with panel_path.open() as fh:
        panel_rows = [normalize_row_dict(r) for r in csv.DictReader(fh, delimiter="\t")]
    panel_ids = {(r.get("variant_id") or "").strip() for r in panel_rows if (r.get("variant_id") or "").strip()}

    # Only keep Janus scores for panel IDs to avoid contamination from stale/extra rows.
    ddg_by_id = {}
    for r in rows:
        vid = (r.get(id_col) or "").strip()
        if not vid or vid not in panel_ids:
            continue
        ddg = safe_float(r.get(ddg_col))
        if ddg is not None:
            ddg_by_id[vid] = ddg

    mix = fit_two_gaussian_mixture(ddg_by_id.values())
    robust_med, robust_sigma = robust_location_scale(ddg_by_id.values())
    pval_pairs = []
    for vid, x in ddg_by_id.items():
        if robust_sigma is None:
            pval = None
        else:
            z = (x - robust_med) / robust_sigma
            # Tail direction depends on score sign convention.
            if args.janus_positive_stabilizing:
                # lower-tail p-value: very negative z => very small p
                pval = normal_cdf(z)
            else:
                # upper-tail p-value when larger ddG is more destabilizing.
                pval = 1.0 - normal_cdf(z)
        pval_pairs.append((vid, pval))
    fdr_outliers = bh_fdr_flags(pval_pairs, q=max(1e-6, min(0.5, args.stability_fdr)))
    bayes_outliers = bayes_posterior_fdr_outliers(
        ddg_by_id,
        mix,
        q=args.stability_fdr,
        janus_positive_stabilizing=args.janus_positive_stabilizing,
    )
    bayes_safeguard_triggered = False
    if ddg_by_id and (len(bayes_outliers) / float(len(ddg_by_id))) > 0.50:
        # Defensive guard against pathological gating on malformed mixtures/scores.
        bayes_outliers = set()
        bayes_safeguard_triggered = True

    enriched = []
    missing = 0
    gated_out = 0
    for r in panel_rows:
        vid = (r.get("variant_id") or "").strip()
        ddg = ddg_by_id.get(vid)
        if ddg is None:
            missing += 1
            stability_score = 0.5
            stability_ok = None
            p_destab = None
            outlier = None
        else:
            if args.stability_gate == "bayes":
                if mix is None:
                    p_destab = None
                    outlier = None
                    stability_ok = None
                    stability_score = 0.5
                else:
                    p_destab = posterior_destabilizing_component(
                        ddg, mix, janus_positive_stabilizing=args.janus_positive_stabilizing
                    )
                    outlier = vid in bayes_outliers
                    stability_ok = not outlier
                    stability_score = 1.0 - p_destab
            else:
                raise RuntimeError(f"Unsupported stability gate: {args.stability_gate}")

        p_func = safe_float(r.get("p_func"))
        p_bind = safe_float(r.get("p_bind"))
        if p_func is None:
            p_func = safe_float(r.get("vespag_shrunk_posterior"))
        if p_bind is None:
            p_bind = 0.5
        if p_func is None:
            p_func = 0.5

        triage_score = 0.45 * p_func + 0.35 * p_bind + 0.20 * stability_score

        out = dict(r)
        out["janus_ddg"] = "" if ddg is None else round(ddg, 6)
        out["janus_stability_score"] = round(stability_score, 6)
        out["janus_stability_ok"] = "" if stability_ok is None else bool(stability_ok)
        out["janus_p_destabilizing"] = "" if p_destab is None else round(float(p_destab), 6)
        out["janus_outlier"] = "" if outlier is None else bool(outlier)
        out["triage_score"] = round(triage_score, 6)
        if args.stability_gate == "bayes" and args.drop_janus_outliers and outlier is True:
            gated_out += 1
            continue
        enriched.append(out)

    enriched.sort(key=lambda x: safe_float(x.get("triage_score")) or 0.0, reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields_out = list(enriched[0].keys()) if enriched else []
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields_out, delimiter="\t")
        writer.writeheader()
        for r in enriched:
            writer.writerow(r)

    ddgs_all = [float(x) for x in ddg_by_id.values() if x is not None and math.isfinite(float(x))]
    ddgs_kept = [safe_float(r.get("janus_ddg")) for r in enriched]
    ddgs_kept = [x for x in ddgs_kept if x is not None]
    summary = {
        "round": args.round,
        "panel_rows": len(panel_rows),
        "janus_matched": len(panel_rows) - missing,
        "janus_missing": missing,
        "janus_gated_out": gated_out,
        "ddg_column": ddg_col,
        "stability_gate": args.stability_gate,
        "stability_fdr": args.stability_fdr,
        "janus_positive_stabilizing": bool(args.janus_positive_stabilizing),
        "ddg_stats_all": {
            "min": min(ddgs_all) if ddgs_all else None,
            "median": (sorted(ddgs_all)[len(ddgs_all)//2] if ddgs_all else None),
            "max": max(ddgs_all) if ddgs_all else None,
        },
        "ddg_stats_kept": {
            "min": min(ddgs_kept) if ddgs_kept else None,
            "median": (sorted(ddgs_kept)[len(ddgs_kept)//2] if ddgs_kept else None),
            "max": max(ddgs_kept) if ddgs_kept else None,
        },
        "robust_null": {
            "median": robust_med,
            "sigma": robust_sigma,
            "fdr_outlier_count": len(fdr_outliers),
            "bayes_tail_fdr_outlier_count": len(bayes_outliers),
            "bayes_outlier_method": "posterior_bfdr",
            "bayes_safeguard_triggered": bool(bayes_safeguard_triggered),
        },
        "mixture": None if mix is None else {
            "low": {
                "pi": round(float(mix["low"]["pi"]), 6),
                "mu": round(float(mix["low"]["mu"]), 6),
                "var": round(float(mix["low"]["var"]), 6),
            },
            "high": {
                "pi": round(float(mix["high"]["pi"]), 6),
                "mu": round(float(mix["high"]["mu"]), 6),
                "var": round(float(mix["high"]["var"]), 6),
            },
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {out_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
