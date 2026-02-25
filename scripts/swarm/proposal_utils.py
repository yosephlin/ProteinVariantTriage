import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def load_site_cards(path: Path) -> List[Dict]:
    cards = []
    if not path.exists():
        return cards
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except Exception:
                continue
    return cards


def compact_site_card(card: Dict) -> Dict:
    tags = card.get("tags") or []
    domains = sorted({t.split(":", 1)[1] for t in tags if t.startswith("domain:")})
    tags = card.get("tags") or []
    ligand_contact = (
        "pdbe_ligand_contact" in tags
        or "pdb_ligand_contact" in tags
        or bool(card.get("ligand_contact"))
    )
    interface = (
        "pdbe_interface" in tags
        or "pdb_interface" in tags
        or ("pdbe_interface" in (card.get("risk_flags") or []))
        or bool(card.get("interface"))
    )
    compact = {
        "chain": card.get("chain"),
        "pos": card.get("pos"),
        "wt": card.get("wt"),
        "numbering": card.get("numbering") or {},
        "tier": card.get("tier"),
        "dist_ligand": card.get("dist_ligand"),
        "dist_functional": card.get("dist_functional"),
        "plddt": card.get("plddt"),
        "exposure": card.get("exposure"),
        "fpocket_member": bool(card.get("fpocket_member")),
        "fpocket_occupancy_confidence": card.get("fpocket_occupancy_confidence"),
        "ss": card.get("ss"),
        "do_not_mutate": bool(card.get("do_not_mutate")),
        "hard_constraints": card.get("hard_constraints") or [],
        "soft_constraints": card.get("soft_constraints") or [],
        "domains": domains,
        "ligand_contact": ligand_contact,
        "interface": interface,
        "functional_site": bool(card.get("functional_site")),
        "functional_reasons": card.get("functional_reasons") or [],
        "buried_core": bool(card.get("buried_core")),
        "structural_lock": bool(card.get("structural_lock")),
        "evolution_conservation": card.get("evolution_conservation"),
        "evolution_allowed_aas": card.get("evolution_allowed_aas") or [],
        "conservation_rank": card.get("conservation_rank"),
        "conservation_top_fraction": card.get("conservation_top_fraction"),
        "site_prefilter_keep": card.get("site_prefilter_keep"),
        "site_prefilter_reasons": card.get("site_prefilter_reasons") or [],
        "site_prefilter_stats": card.get("site_prefilter_stats") or {},
    }
    prolif = card.get("prolif") or {}
    if prolif:
        compact["prolif"] = {
            "contact_freq": prolif.get("contact_freq"),
            "top_interactions": (prolif.get("top_interactions") or [])[:3],
        }
    return compact


def plddt_bin(plddt: Optional[float], low: float = 70.0, high: float = 90.0) -> str:
    if plddt is None:
        return "unknown"
    if plddt < low:
        return "low"
    if plddt < high:
        return "mid"
    return "high"


def _domain_label(card: Dict) -> str:
    doms = card.get("domains") or []
    return doms[0] if doms else "none"


def stratified_sample(
    cards: List[Dict],
    num_sites: int,
    seed: int = 13,
    include_do_not_mutate: bool = False,
    tier_weights: Optional[Dict[int, float]] = None,
    plddt_low: float = 70.0,
    plddt_high: float = 90.0,
) -> List[Dict]:
    pool = [c for c in cards if include_do_not_mutate or not c.get("do_not_mutate")]
    if num_sites <= 0 or num_sites >= len(pool):
        return list(pool)

    rng = random.Random(seed)
    strata: Dict[Tuple[int, str, str], List[Dict]] = {}
    for c in pool:
        key = (int(c.get("tier") or 0), _domain_label(c), plddt_bin(c.get("plddt"), plddt_low, plddt_high))
        strata.setdefault(key, []).append(c)
    for lst in strata.values():
        rng.shuffle(lst)

    # If there are more strata than desired sites, sample strata directly.
    if len(strata) >= num_sites:
        keys = list(strata.keys())
        rng.shuffle(keys)
        selected = []
        for k in keys[:num_sites]:
            if strata[k]:
                selected.append(strata[k].pop())
        return selected

    # Coverage-first: round-robin 1 per stratum.
    selected = []
    keys = sorted(strata.keys(), key=lambda k: (k[0], k[1], k[2]))
    while keys and len(selected) < num_sites:
        picked = False
        for k in list(keys):
            if len(selected) >= num_sites:
                break
            lst = strata[k]
            if lst:
                selected.append(lst.pop())
                picked = True
            if not lst:
                keys.remove(k)
        if not picked:
            break

    # Fill remainder with tier-weighted sampling.
    remaining = []
    for lst in strata.values():
        remaining.extend(lst)
    if not remaining or len(selected) >= num_sites:
        return selected

    if tier_weights is None:
        tier_weights = {}
    tiers = {}
    for c in remaining:
        tiers.setdefault(int(c.get("tier") or 0), []).append(c)
    for lst in tiers.values():
        rng.shuffle(lst)

    while len(selected) < num_sites and any(tiers.values()):
        available = [t for t, lst in tiers.items() if lst]
        weights = [tier_weights.get(t, 1.0) for t in available]
        if sum(weights) <= 0:
            weights = [1.0 for _ in available]
        choice = rng.choices(available, weights=weights, k=1)[0]
        selected.append(tiers[choice].pop())

    return selected


def shard_sites(cards: List[Dict], shard_size: int) -> List[List[Dict]]:
    if shard_size <= 0:
        return [list(cards)]
    return [cards[i:i + shard_size] for i in range(0, len(cards), shard_size)]


def summarize_context(context_pack: Dict) -> Dict:
    if not context_pack:
        return {}
    target = context_pack.get("target") or {}
    ligand = context_pack.get("ligand") or {}
    pocket = context_pack.get("pocket") or {}
    rules = context_pack.get("rules") or {}
    api = context_pack.get("api_context") or {}
    chembl = context_pack.get("chembl") or api.get("chembl") or {}
    ligand_smiles = ligand.get("smiles")
    summary = {
        "uniprot_id": target.get("uniprot_id"),
        "length": target.get("length"),
        "chains": target.get("chains_present", []),
        "pocket_residue_count": len(pocket.get("pocket_residues_chain") or []),
        "primary_ligand_scores": (ligand.get("vina_gnina_summary") or {}).get("primary_scores") or {},
        "rules": rules,
    }
    if ligand_smiles:
        summary["ligand_features"] = summarize_smiles(ligand_smiles)
        summary["ligand_smiles"] = ligand_smiles
    if isinstance(chembl, dict) and chembl:
        target = chembl.get("target") if isinstance(chembl.get("target"), dict) else {}
        if target:
            summary["chembl_target"] = {
                "target_chembl_id": target.get("target_chembl_id"),
                "pref_name": target.get("pref_name"),
                "organism": target.get("organism"),
                "target_type": target.get("target_type"),
            }
        note = chembl.get("note")
        if note:
            summary["chembl_note"] = note
        priors = chembl.get("ligand_priors") if isinstance(chembl.get("ligand_priors"), list) else []
        summary["chembl_prior_count"] = int(len(priors))
        if priors:
            summary["chembl_top_ligands"] = priors[:5]
    return summary


def _pick_smiles(smiles) -> Optional[str]:
    if isinstance(smiles, str):
        return smiles
    if isinstance(smiles, dict):
        for k in ("canonical_smiles", "isomeric_smiles", "SMILES", "ConnectivitySMILES"):
            v = smiles.get(k)
            if v:
                return v
    return None


def summarize_smiles(smiles) -> Dict:
    raw = _pick_smiles(smiles)
    if not raw:
        return {"note": "missing_smiles"}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except Exception:
        return {"note": "rdkit_unavailable"}

    try:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return {"note": "invalid_smiles"}
        return {
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": Lipinski.RingCount(mol),
            "aromatic_rings": Lipinski.NumAromaticRings(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
        }
    except Exception:
        return {"note": "rdkit_error"}


def parse_tier_weights(raw: Optional[str]) -> Optional[Dict[int, float]]:
    if not raw:
        return None
    weights: Dict[int, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            weights[int(k.strip())] = float(v.strip())
        except Exception:
            continue
    return weights or None
