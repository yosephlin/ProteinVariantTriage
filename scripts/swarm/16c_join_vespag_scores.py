import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from artifact_paths import proposals_path as round_proposals_path, proposals_vespag_path, vespag_scores_csv_path
except ImportError:
    from scripts.swarm.artifact_paths import proposals_path as round_proposals_path, proposals_vespag_path, vespag_scores_csv_path

try:
    from mutation_utils import mutations_to_id, row_mutations
except ImportError:
    from scripts.swarm.mutation_utils import mutations_to_id, row_mutations


def proposal_mutation_id(p: Dict) -> str:
    return mutations_to_id(row_mutations(p), include_chain=False).upper()


def proposal_component_mutation_ids(p: Dict) -> List[str]:
    mids: List[str] = []
    for m in row_mutations(p):
        tok = mutations_to_id([m], include_chain=False).upper()
        if tok:
            mids.append(tok)
    return mids


def proposal_positions(p: Dict) -> List[int]:
    muts = row_mutations(p)
    out: List[int] = []
    for m in muts:
        try:
            out.append(int(m["pos"]))
        except Exception:
            continue
    return out


def _geometric_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    vals = [float(max(1e-8, min(1.0, v))) for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in vals) / float(len(vals))))


def resolve_proposal_score(
    p: Dict,
    score_map: Dict[str, float],
    post_map: Dict[str, float],
) -> Tuple[Optional[float], Optional[float], str]:
    mid = proposal_mutation_id(p)
    s = score_map.get(mid)
    pp = post_map.get(mid)
    if s is not None and pp is not None:
        return float(s), float(pp), "direct"

    comp_ids = proposal_component_mutation_ids(p)
    if len(comp_ids) <= 1:
        return None, None, "missing"

    comp_scores = [score_map.get(cid) for cid in comp_ids]
    comp_posts = [post_map.get(cid) for cid in comp_ids]
    if any(v is None for v in comp_scores) or any(v is None for v in comp_posts):
        return None, None, "missing_components"

    comp_scores_f = [float(v) for v in comp_scores if v is not None]
    comp_posts_f = [float(v) for v in comp_posts if v is not None]
    if not comp_scores_f or not comp_posts_f:
        return None, None, "missing_components"

    # Multi-point function proxy: arithmetic mean on normalized score, geometric
    # mean on posterior to preserve a conservative compounded viability signal.
    s_comp = float(np.mean(np.asarray(comp_scores_f, dtype=float)))
    pp_comp = _geometric_mean(comp_posts_f)
    if not math.isfinite(pp_comp):
        return None, None, "missing_components"
    return float(clamp(s_comp, 0.0, 1.0)), float(clamp(pp_comp, 0.0, 1.0)), "composed_from_singles"


def load_proposals(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def pick_column(fieldnames: List[str], preferred: List[str], contains: List[str]) -> Optional[str]:
    lowers = {f.lower(): f for f in fieldnames}
    for p in preferred:
        if p in lowers:
            return lowers[p]
    for f in fieldnames:
        lf = f.lower()
        if any(c in lf for c in contains):
            return f
    return None


def detect_score_column(rows: List[Dict[str, str]], fieldnames: List[str]) -> Optional[str]:
    blocked = ("pos", "position", "index", "protein", "mutation", "variant", "id")
    candidates: List[str] = []
    for f in fieldnames:
        lf = f.lower()
        if any(b in lf for b in blocked):
            continue
        if "score" in lf or "pred" in lf or "effect" in lf or "vespag" in lf:
            candidates.append(f)
    if not candidates:
        return None

    for f in candidates:
        for r in rows[:20]:
            v = r.get(f)
            if v is None:
                continue
            try:
                float(v)
                return f
            except Exception:
                continue
    return candidates[0]


def safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_minmax(vals: np.ndarray) -> np.ndarray:
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        return np.full_like(vals, 0.5)
    return (vals - lo) / (hi - lo)


def beta_params_from_weighted_moments(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    ww = np.maximum(w, eps)
    ww = ww / np.sum(ww)
    mean = float(np.sum(ww * x))
    var = float(np.sum(ww * ((x - mean) ** 2)))

    mean = min(1.0 - eps, max(eps, mean))
    # Keep variance in feasible beta range.
    var = min(max(var, 1e-5), mean * (1.0 - mean) - 1e-5)

    kappa = (mean * (1.0 - mean) / var) - 1.0
    kappa = min(max(kappa, 1.0), 1000.0)
    a = mean * kappa
    b = (1.0 - mean) * kappa
    return float(max(0.2, min(500.0, a))), float(max(0.2, min(500.0, b)))


def log_beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) + log_norm


def fit_beta_mixture_posteriors(x: np.ndarray, iters: int = 80) -> Tuple[np.ndarray, Dict[str, float]]:
    # x expected in [0, 1]
    x = np.clip(x.astype(float), 1e-6, 1.0 - 1e-6)
    n = len(x)
    if n < 8:
        return np.full(n, 0.5), {
            "pi_low": 0.5,
            "pi_high": 0.5,
            "a_low": 2.0,
            "b_low": 5.0,
            "a_high": 5.0,
            "b_high": 2.0,
        }

    # Initialize by quantiles for stable EM start.
    med = float(np.median(x))
    r_high = (x >= med).astype(float)
    r_low = 1.0 - r_high
    pi_high = max(0.1, min(0.9, float(np.mean(r_high))))
    pi_low = 1.0 - pi_high
    a_low, b_low = beta_params_from_weighted_moments(x, np.maximum(r_low, 1e-3))
    a_high, b_high = beta_params_from_weighted_moments(x, np.maximum(r_high, 1e-3))

    for _ in range(max(5, iters)):
        l_low = np.log(max(1e-8, pi_low)) + log_beta_pdf(x, a_low, b_low)
        l_high = np.log(max(1e-8, pi_high)) + log_beta_pdf(x, a_high, b_high)
        mx = np.maximum(l_low, l_high)
        den = np.exp(l_low - mx) + np.exp(l_high - mx)
        r_high = np.exp(l_high - mx) / np.maximum(1e-12, den)
        r_low = 1.0 - r_high

        pi_high = float(np.mean(r_high))
        pi_high = max(0.05, min(0.95, pi_high))
        pi_low = 1.0 - pi_high
        a_low, b_low = beta_params_from_weighted_moments(x, r_low)
        a_high, b_high = beta_params_from_weighted_moments(x, r_high)

    mean_low = a_low / (a_low + b_low)
    mean_high = a_high / (a_high + b_high)

    if mean_high < mean_low:
        # swap components so "high" is viability component
        pi_low, pi_high = pi_high, pi_low
        a_low, a_high = a_high, a_low
        b_low, b_high = b_high, b_low

    l_low = np.log(max(1e-8, pi_low)) + log_beta_pdf(x, a_low, b_low)
    l_high = np.log(max(1e-8, pi_high)) + log_beta_pdf(x, a_high, b_high)
    mx = np.maximum(l_low, l_high)
    den = np.exp(l_low - mx) + np.exp(l_high - mx)
    post_high = np.exp(l_high - mx) / np.maximum(1e-12, den)

    return post_high, {
        "pi_low": round(float(pi_low), 6),
        "pi_high": round(float(pi_high), 6),
        "a_low": round(float(a_low), 6),
        "b_low": round(float(b_low), 6),
        "a_high": round(float(a_high), 6),
        "b_high": round(float(b_high), 6),
    }


def efdr_threshold(p: np.ndarray, alpha: float, min_p: float) -> float:
    if p.size == 0:
        return 1.1
    ps = np.sort(np.clip(p, 0.0, 1.0))[::-1]
    false_cum = np.cumsum(1.0 - ps)
    denom = np.arange(1, len(ps) + 1, dtype=float)
    efdr = false_cum / denom
    idx = np.where((efdr <= alpha) & (ps >= min_p))[0]
    if len(idx) == 0:
        return 1.1
    return float(ps[idx[-1]])


def rank_quantile(values: List[float], v: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return 1.0
    lo, hi = 0, len(vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if vals[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    rank = max(0, lo - 1)
    return rank / (len(vals) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Join VespaG scores and compute Bayesian tri-band gating.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--proposals", default=None)
    ap.add_argument("--vespag-csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--protein-id", default=None)
    ap.add_argument("--auto-normalize", action="store_true", default=True)
    ap.add_argument("--no-auto-normalize", dest="auto_normalize", action="store_false")
    ap.add_argument("--shrinkage-prior", type=float, default=10.0, help="Empirical-Bayes shrinkage strength for site means.")
    ap.add_argument("--fdr-green", type=float, default=0.12, help="Expected FDR bound for green band.")
    ap.add_argument("--fdr-soft", type=float, default=0.30, help="Expected FDR bound for soft pass (green+amber).")
    ap.add_argument("--min-soft-posterior", type=float, default=0.40)
    ap.add_argument("--green-z", type=float, default=0.5, help="Distributional z-threshold fallback for green when strict FDR is infeasible.")
    ap.add_argument("--mixture-iters", type=int, default=80)
    ap.add_argument("--contact-rescue", action="store_true", default=True,
                    help="Promote some red-band contact-centric mutations to amber when function posterior is acceptable.")
    ap.add_argument("--no-contact-rescue", dest="contact_rescue", action="store_false",
                    help="Disable contact-aware amber rescue.")
    ap.add_argument("--contact-rescue-min-posterior", type=float, default=0.35,
                    help="Minimum shrunk posterior for contact-aware red->amber rescue.")
    ap.add_argument("--contact-rescue-prolif-threshold", type=float, default=0.20,
                    help="Minimum ProLIF contact frequency for rescue.")
    ap.add_argument("--contact-rescue-max-dist", type=float, default=4.0,
                    help="Ligand-distance cutoff (Angstrom) for rescue when direct contact is present.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    proposals_path = Path(args.proposals) if args.proposals else round_proposals_path(outdir=outdir, round_id=int(args.round))
    if args.vespag_csv:
        vespag_csv = Path(args.vespag_csv)
    else:
        vespag_csv = vespag_scores_csv_path(outdir=outdir, round_id=int(args.round))
    out_path = Path(args.out) if args.out else proposals_vespag_path(outdir=outdir, round_id=int(args.round))

    proposals = load_proposals(proposals_path)
    if not proposals:
        raise SystemExit(f"No proposals found at {proposals_path}")
    if not vespag_csv.exists():
        raise SystemExit(f"VespaG CSV not found: {vespag_csv}")

    with vespag_csv.open() as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    mut_col = pick_column(fieldnames, ["mutation_id"], ["mutation", "variant", "mut"])
    if not mut_col:
        raise SystemExit("Could not find mutation id column in VespaG CSV.")
    score_col = detect_score_column(rows, fieldnames)
    if not score_col:
        raise SystemExit("Could not infer score column in VespaG CSV.")
    prot_col = pick_column(fieldnames, ["protein_id"], ["protein", "id"])

    score_map: Dict[str, float] = {}
    for r in rows:
        if prot_col and args.protein_id:
            if (r.get(prot_col) or "").strip() != args.protein_id:
                continue
        mid = (r.get(mut_col) or "").strip().upper()
        if not mid:
            continue
        sc = safe_float(r.get(score_col), float("nan"))
        if not math.isfinite(sc):
            continue
        score_map[mid] = sc

    if not score_map:
        raise SystemExit("No VespaG scores were mapped to mutation ids.")

    all_scores = np.array(list(score_map.values()), dtype=float)
    if args.auto_normalize and (float(np.min(all_scores)) < 0.0 or float(np.max(all_scores)) > 1.0):
        all_scores = normalize_minmax(all_scores)
        for i, k in enumerate(list(score_map.keys())):
            score_map[k] = float(all_scores[i])
    else:
        all_scores = np.clip(all_scores, 0.0, 1.0)
        for i, k in enumerate(list(score_map.keys())):
            score_map[k] = float(all_scores[i])

    # Global posterior p(functional | score) via Beta mixture.
    score_vals = np.array(list(score_map.values()), dtype=float)
    p_post, mix_params = fit_beta_mixture_posteriors(score_vals, iters=args.mixture_iters)
    post_map: Dict[str, float] = {}
    for i, k in enumerate(list(score_map.keys())):
        post_map[k] = float(p_post[i])

    # Site-level empirical-Bayes shrinkage on posterior means.
    site_post: Dict[int, List[float]] = defaultdict(list)
    for p in proposals:
        pos_list = proposal_positions(p)
        if not pos_list:
            continue
        _s, pp, _src = resolve_proposal_score(p=p, score_map=score_map, post_map=post_map)
        if pp is None:
            continue
        for pos in pos_list:
            site_post[pos].append(float(pp))

    site_mean = {pos: float(np.mean(vals)) for pos, vals in site_post.items() if vals}
    global_mean = float(np.mean(list(post_map.values())))

    shrunk_values: List[float] = []
    for p in proposals:
        pos_list = proposal_positions(p)
        if not pos_list:
            continue
        mid = proposal_mutation_id(p)
        pp = post_map.get(mid)
        if pp is None:
            continue
        n_site = int(np.mean([len(site_post.get(pos, [])) for pos in pos_list]))
        lam = float(args.shrinkage_prior) / (float(args.shrinkage_prior) + float(max(0, n_site)))
        smean = float(np.mean([site_mean.get(pos, global_mean) for pos in pos_list]))
        p_shrunk = lam * pp + (1.0 - lam) * smean
        shrunk_values.append(p_shrunk)

    shrunk_np = np.array(shrunk_values, dtype=float) if shrunk_values else np.array([], dtype=float)
    t_green = efdr_threshold(shrunk_np, alpha=args.fdr_green, min_p=max(args.min_soft_posterior, 0.5))
    t_soft = efdr_threshold(shrunk_np, alpha=args.fdr_soft, min_p=args.min_soft_posterior)
    # Stabilize thresholds for small-N rounds where eFDR can return >1.0.
    if t_green > 1.0:
        if shrunk_np.size:
            mu = float(np.mean(shrunk_np))
            sig = float(np.std(shrunk_np))
            t_green = clamp(mu + max(0.0, args.green_z) * sig, max(0.55, args.min_soft_posterior + 0.05), 0.999)
        else:
            t_green = 0.999
    if t_soft > 1.0:
        if shrunk_np.size:
            q = float(np.quantile(shrunk_np, 0.45))
            t_soft = clamp(q, args.min_soft_posterior, max(args.min_soft_posterior, t_green - 0.02))
        else:
            t_soft = max(args.min_soft_posterior, 0.40)
    if t_soft > t_green:
        t_soft = max(args.min_soft_posterior, t_green - 0.02)

    green = amber = red = missing = rescued = 0
    with out_path.open("w") as out_fh:
        for p in proposals:
            pos_list = proposal_positions(p)
            s, pp, score_source = resolve_proposal_score(p=p, score_map=score_map, post_map=post_map)
            if s is None or pp is None:
                p["vespag_score_norm"] = None
                p["vespag_posterior"] = None
                p["vespag_shrunk_posterior"] = None
                p["vespag_global_quantile"] = None
                p["vespag_gate_band"] = "red"
                p["vespag_gate_pass"] = False
                p["vespag_strict_pass"] = False
                p["vespag_gate_reason"] = "missing_score"
                p["vespag_score_source"] = "missing"
                missing += 1
                red += 1
                out_fh.write(json.dumps(p, ensure_ascii=False) + "\n")
                continue

            n_site = int(np.mean([len(site_post.get(pos, [])) for pos in pos_list])) if pos_list else 0
            lam = float(args.shrinkage_prior) / (float(args.shrinkage_prior) + float(max(0, n_site)))
            smean = float(np.mean([site_mean.get(pos, global_mean) for pos in pos_list])) if pos_list else global_mean
            p_shrunk = lam * pp + (1.0 - lam) * smean

            if p_shrunk >= t_green:
                band = "green"
                reason = f"posterior_ge_green_{t_green:.4f}"
            elif p_shrunk >= t_soft:
                band = "amber"
                reason = f"posterior_ge_soft_{t_soft:.4f}"
            else:
                band = "red"
                reason = f"posterior_lt_soft_{t_soft:.4f}"

            # Contact-aware rescue: keep some binding-relevant variants for downstream selector.
            if (
                args.contact_rescue
                and band == "red"
                and p_shrunk >= args.contact_rescue_min_posterior
            ):
                prolif_freq = safe_float(p.get("prolif_contact_freq"), 0.0)
                ligand_contact = bool(p.get("ligand_contact"))
                dist_ligand = safe_float(p.get("dist_ligand"), 999.0)
                critical = bool(p.get("critical"))
                near_direct = ligand_contact and (dist_ligand <= args.contact_rescue_max_dist)
                prolif_direct = prolif_freq >= args.contact_rescue_prolif_threshold
                if (prolif_direct or near_direct) and not critical:
                    band = "amber"
                    reason = (
                        f"contact_rescue(p={p_shrunk:.4f},"
                        f"prolif={prolif_freq:.3f},dist={dist_ligand:.2f})"
                    )
                    rescued += 1

            gq = rank_quantile(list(score_map.values()), s)
            p["vespag_score_norm"] = round(float(s), 6)
            p["vespag_posterior"] = round(float(pp), 6)
            p["vespag_shrunk_posterior"] = round(float(p_shrunk), 6)
            p["vespag_global_quantile"] = round(float(gq), 6)
            p["vespag_score_source"] = str(score_source)
            p["vespag_gate_band"] = band
            p["vespag_gate_pass"] = band in ("green", "amber")
            p["vespag_strict_pass"] = band == "green"
            p["vespag_gate_reason"] = reason

            if band == "green":
                green += 1
            elif band == "amber":
                amber += 1
            else:
                red += 1

            out_fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")
    print(
        "[vespag] "
        f"green={green} amber={amber} red={red} missing={missing} total={len(proposals)} "
        f"t_green={t_green:.4f} t_soft={t_soft:.4f} global_mean={global_mean:.4f}"
    )
    if rescued:
        print(f"[vespag] contact_rescued={rescued}")
    print(f"[vespag] mixture={json.dumps(mix_params, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
