import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

try:
    from artifact_paths import panel_path, proposals_vespag_path
except ImportError:
    from scripts.swarm.artifact_paths import panel_path, proposals_vespag_path

try:
    from mutation_utils import row_mutations, row_variant_id
except ImportError:
    from scripts.swarm.mutation_utils import row_mutations, row_variant_id


AA_CLASSES = {
    "hydrophobic": set("AVLIMFWY"),
    "aromatic": set("FYW"),
    "polar": set("STNQC"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "special": set("GP"),
}


def aa_class(aa: str) -> str:
    for k, v in AA_CLASSES.items():
        if aa in v:
            return k
    return "other"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x


def load_panel_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open() as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            if not isinstance(row, dict):
                continue
            cleaned = {}
            for k, v in row.items():
                key = str(k or "").strip()
                if not key:
                    continue
                cleaned[key] = str(v or "").strip()
            vid = str(cleaned.get("variant_id") or "").strip().upper()
            if vid:
                out[vid] = cleaned
    return out


def derive_target_signature(outdir: Path) -> str:
    context_pack = _safe_json(outdir / "swarm" / "context_pack.json")
    target = context_pack.get("target") or {}
    ligand = context_pack.get("ligand") or {}
    structures = context_pack.get("structures") or {}
    inputs = context_pack.get("inputs") or {}
    payload = {
        "outdir": str(outdir.resolve()),
        "uniprot_id": target.get("uniprot_id"),
        "target_length": target.get("length"),
        "ligand_smiles": (ligand.get("smiles") or {}).get("canonical"),
        "receptor_pdb_path": structures.get("receptor_pdb_path") or inputs.get("receptor_pdb_path"),
    }
    fasta = outdir / "enzyme_wt.fasta"
    if fasta.exists():
        payload["enzyme_wt_sha1"] = _file_sha1(fasta)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Update VespaG policy state from gated proposals.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--proposals", default=None)
    ap.add_argument("--panel", default=None, help="Default: OUTDIR/swarm/swarm_panel_rK.tsv")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--target", type=float, default=0.6)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument(
        "--selected-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update policy from selected panel variants only (recommended).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    proposals_path = Path(args.proposals) if args.proposals else proposals_vespag_path(outdir=outdir, round_id=int(args.round))
    panel_rows_path = Path(args.panel) if args.panel else panel_path(outdir=outdir, round_id=int(args.round))
    policy_path = Path(args.policy) if args.policy else outdir / "swarm" / "vespag_policy_state.json"

    items = []
    with proposals_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue

    if policy_path.exists():
        try:
            policy = json.loads(policy_path.read_text())
        except Exception:
            policy = {}
    else:
        policy = {}

    policy.setdefault("version", 1)
    policy.setdefault("sites", {})
    policy.setdefault("meta", {})
    sites: Dict[str, Dict] = policy["sites"]
    meta: Dict[str, str] = policy["meta"]
    target_signature = derive_target_signature(outdir)
    prev_target_signature = str(meta.get("target_signature") or "")
    if prev_target_signature and prev_target_signature != target_signature:
        # Reset per-site policy when switching to a different protein/ligand target.
        sites.clear()
        meta["last_update_signature"] = ""
    meta["target_signature"] = target_signature

    # Idempotency guard: repeated reruns with unchanged proposals should not re-apply updates.
    sig_input = (
        f"round={int(args.round)}|path={str(proposals_path.resolve())}"
        f"|panel={str(panel_rows_path.resolve())}"
        f"|selected_only={int(bool(args.selected_only))}"
    )
    sig = hashlib.sha1(sig_input.encode("utf-8")).hexdigest()
    if proposals_path.exists():
        sig = hashlib.sha1((sig + "|" + hashlib.sha1(proposals_path.read_bytes()).hexdigest()).encode("utf-8")).hexdigest()
    if panel_rows_path.exists():
        sig = hashlib.sha1((sig + "|" + hashlib.sha1(panel_rows_path.read_bytes()).hexdigest()).encode("utf-8")).hexdigest()
    if str(meta.get("last_update_signature") or "") == sig:
        print(f"[vespag-policy] unchanged evidence signature; skipping update: {policy_path}")
        return 0

    panel_rows = load_panel_rows(panel_rows_path)
    selected_seen = 0
    considered_total = 0

    class_accept = Counter()
    class_total = Counter()

    for p in items:
        muts = row_mutations(p)
        if not muts:
            continue
        vid = row_variant_id({**p, "mutations": muts}).upper()
        selected_row = panel_rows.get(vid)
        if bool(args.selected_only) and panel_rows and selected_row is None:
            continue

        if selected_row is not None:
            selected_seen += 1
            bind = clamp(safe_float(selected_row.get("p_bind"), safe_float(p.get("p_bind"), 0.0)), 0.0, 1.0)
            stability = clamp(safe_float(selected_row.get("p_stability"), safe_float(p.get("p_stability"), 0.5)), 0.0, 1.0)
            plaus = clamp(
                safe_float(
                    selected_row.get("p_plausibility"),
                    safe_float(p.get("seq_prior_ensemble_plausibility"), 0.5),
                ),
                0.0,
                1.0,
            )
            retention = clamp(
                safe_float(selected_row.get("prolif_retention"), safe_float(p.get("prolif_retention"), 0.0)),
                0.0,
                1.0,
            )
            accepted_weight = clamp(
                (0.50 * bind) + (0.20 * stability) + (0.20 * plaus) + (0.10 * retention),
                0.0,
                1.0,
            )
        else:
            bind = clamp(safe_float(p.get("p_bind"), 0.0), 0.0, 1.0)
            stability = clamp(safe_float(p.get("p_stability"), 0.5), 0.0, 1.0)
            plaus = clamp(safe_float(p.get("seq_prior_ensemble_plausibility"), 0.5), 0.0, 1.0)
            retention = clamp(safe_float(p.get("prolif_retention"), 0.0), 0.0, 1.0)
            orth = clamp((0.55 * bind) + (0.20 * stability) + (0.20 * plaus) + (0.05 * retention), 0.0, 1.0)
            accepted_weight = clamp(0.25 * orth, 0.0, 1.0)

        accepted_weight = clamp(accepted_weight, 0.0, 1.0)
        for m in muts:
            chain = str(m.get("chain") or "A")
            try:
                pos_i = int(m.get("pos"))
            except Exception:
                continue
            wt = str(m.get("wt") or "").upper()
            mut = str(m.get("mut") or "").upper()
            if not wt or not mut:
                continue

            key = f"{chain}:{pos_i}"
            site = sites.setdefault(key, {"wt": wt, "aa": {}})
            aa = site["aa"].setdefault(mut, {"proposed": 0.0, "accepted": 0.0, "rejected": 0.0, "logit": 0.0})
            considered_total += 1
            aa["proposed"] += 1
            aa["accepted"] += accepted_weight
            aa["rejected"] += (1.0 - accepted_weight)

            cls = aa_class(mut)
            class_total[cls] += 1
            class_accept[cls] += accepted_weight

    for site in sites.values():
        for mut, stats in site.get("aa", {}).items():
            proposed = stats.get("proposed", 0)
            accepted = stats.get("accepted", 0)
            if proposed <= 0:
                continue
            acc_rate = accepted / proposed
            delta = args.alpha * (acc_rate - args.target)
            stats["logit"] = clamp(stats.get("logit", 0.0) + delta, -args.clip, args.clip)
            stats["accept_rate"] = round(acc_rate, 4)

    policy["summary"] = {
        "total_mutations": len(items),
        "policy_updates_considered": int(considered_total),
        "policy_updates_from_selected": int(selected_seen),
        "update_source": "selected_panel_only" if bool(args.selected_only) else "selected_plus_weak_unsampled",
        "class_accept_rate": {
            k: round(class_accept[k] / class_total[k], 4) if class_total[k] else 0.0
            for k in sorted(class_total)
        },
        "underexplored_classes": [k for k, _ in class_total.most_common()][-2:],
    }
    meta["last_update_signature"] = sig
    meta["last_update_round"] = int(args.round)

    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2))
    print(f"Wrote: {policy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
