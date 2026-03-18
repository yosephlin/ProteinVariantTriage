from pathlib import Path


def swarm_root(outdir: Path) -> Path:
    return Path(outdir) / "swarm"


def round_tag(round_id: int) -> str:
    return f"r{int(round_id)}"


def round_artifact(outdir: Path, round_id: int, stem: str, suffix: str) -> Path:
    return swarm_root(outdir) / f"{stem}_{round_tag(round_id)}{suffix}"


def proposals_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "proposals", ".jsonl")


def proposals_vespag_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "proposals_vespag", ".jsonl")


def panel_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "swarm_panel", ".tsv")


def panel_summary_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "swarm_panel_summary", ".json")


def panel_with_janus_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "swarm_panel_with_janus", ".tsv")


def panel_with_janus_summary_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "swarm_panel_with_janus_summary", ".json")


def round_manifest_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "manifest", ".json")


def round_diagnostics_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "stat_model_diagnostics", ".json")


def recursive_iteration_metrics_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "recursive_iteration_metrics", ".json")


def binding_fastdl_summary_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "binding_fastdl_summary", ".json")


def binding_fastdl_cache_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "binding_fastdl_cache", ".json")


def binding_fastdl_mutants_dir(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "binding_fastdl_mutants", "")


def vespag_mutation_csv_path(outdir: Path, round_id: int) -> Path:
    return swarm_root(outdir) / f"vespag_round{int(round_id)}_mutations.csv"


def vespag_scores_csv_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "vespag_scores", ".csv")


def janus_input_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "janus_input", ".csv")


def janus_scores_path(outdir: Path, round_id: int) -> Path:
    return round_artifact(outdir, round_id, "janus_scores", ".csv")


def final_swarm_panel_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_panel.tsv"


def final_swarm_panel_summary_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_panel_summary.json"


def final_swarm_panel_production_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_panel_production.tsv"


def final_swarm_panel_explore_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_panel_explore.tsv"


def final_janus_input_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "janus_input.csv"


def final_janus_scores_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "janus_scores.csv"


def final_with_janus_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_with_janus.tsv"


def final_with_janus_summary_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_with_janus_summary.json"


def final_with_janus_production_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_with_janus_production.tsv"


def final_with_janus_production_summary_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_final_with_janus_production_summary.json"


def minimal_af2_panel_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_minimal_af2_panel.tsv"


def minimal_af2_panel_vespag_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_minimal_af2_panel_vespag.tsv"


def minimal_af2_summary_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_minimal_af2_summary.json"


def minimal_af2_candidates_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_minimal_af2_candidates.jsonl"


def minimal_af2_vespag_mutation_csv_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_minimal_af2_vespag_mutations.csv"


def minimal_af2_vespag_scores_csv_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_minimal_af2_vespag_scores.csv"


def minimal_af2_vespag_summary_path(outdir: Path) -> Path:
    return swarm_root(outdir) / "swarm_minimal_af2_vespag_summary.json"
