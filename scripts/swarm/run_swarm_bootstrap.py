import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from input_paths import infer_uniprot_accession, resolve_canonical_fasta
except ImportError:
    from scripts.swarm.input_paths import infer_uniprot_accession, resolve_canonical_fasta


def run(cmd, env=None):
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SWARM local bootstrap (API + local packs)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--skip-api", action="store_true")
    ap.add_argument("--force-api", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-prolif", action="store_true", help="Skip ProLIF enrichment of site cards.")
    ap.add_argument("--prolif-max-poses", type=int, default=20, help="Max docking poses to include in ProLIF enrichment.")
    ap.add_argument("--skip-input-prep", action="store_true", help="Skip canonical input preparation (14a).")

    # Input preparation
    ap.add_argument("--input-spec", default=None, help="JSON specification for protein/ligand/fasta sources.")
    ap.add_argument("--protein-source", default=None)
    ap.add_argument("--protein-path", default=None)
    ap.add_argument("--protein-id", default=None, help="UniProt accession (AF) or PDB id (RCSB).")
    ap.add_argument("--protein-chain", default=None)
    ap.add_argument("--fasta-source", default=None)
    ap.add_argument("--fasta-path", default=None)
    ap.add_argument("--fasta-accession", default=None)
    ap.add_argument("--ligand-source", default=None)
    ap.add_argument("--ligand-path", default=None)
    ap.add_argument("--ligand-id", default=None)
    ap.add_argument("--ligand-smiles", default=None)

    # UniProt resolution
    ap.add_argument("--accession", default=None)
    ap.add_argument("--protein-name", default=None)
    ap.add_argument("--organism-id", default=None)
    ap.add_argument("--organism-name", default=None)
    ap.add_argument("--reviewed-only", action="store_true")

    # Optional enrichments
    ap.add_argument("--include-chembl", action="store_true")
    ap.add_argument("--include-hmmer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-mcsa", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--chembl-query", default=None)
    ap.add_argument("--chembl-organism", default=None)
    ap.add_argument("--ligand-name", default=None)
    ap.add_argument("--ligand-cid", default=None)
    ap.add_argument("--evmutation-priors", default=None, help="Optional EVmutation score table to build sequence priors.")
    ap.add_argument("--eve-priors", default=None, help="Optional EVE score table to build sequence priors.")
    ap.add_argument("--deepsequence-priors", default=None, help="Optional DeepSequence score table to build sequence priors.")

    # Single recursive execution
    ap.add_argument("--run-recursive", action=argparse.BooleanOptionalAction, default=True,
                    help="Run the single recursive SWARM loop after bootstrap prep.")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--focus-round", type=int, default=None)
    ap.add_argument("--panel-total", type=int, default=200)
    ap.add_argument("--global-panel-budget", type=int, default=0)
    ap.add_argument("--final-max-candidates", type=int, default=0)
    ap.add_argument("--proposal-total", type=int, default=0)
    ap.add_argument("--proposal-max-per-position", type=int, default=4)
    ap.add_argument("--max-mutations-per-variant", type=int, default=2)
    ap.add_argument("--multi-point-fraction", type=float, default=0.35)
    ap.add_argument("--multi-seed-size", type=int, default=120)
    ap.add_argument("--multi-max-candidates", type=int, default=1200)
    ap.add_argument("--multi-min-position-separation", type=int, default=1)
    ap.add_argument("--multi-max-position-separation", type=int, default=0)
    ap.add_argument("--critical-blosum-min", type=int, default=-1)
    ap.add_argument("--strict-evo-conservation-threshold", type=float, default=0.95)
    ap.add_argument("--strict-evo-blosum-min", type=int, default=-1)
    ap.add_argument("--functional-exploratory-enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--functional-exploratory-blosum-min", type=int, default=-2)
    ap.add_argument("--functional-exploratory-max-extra", type=int, default=3)
    ap.add_argument("--functional-exploratory-ligand-shell", type=float, default=8.0)
    ap.add_argument("--functional-site-hard-filter", action="store_true", default=False)
    ap.add_argument("--near-functional-hard-filter", action="store_true", default=False)
    ap.add_argument("--site-card-wt-mismatch-max-frac", type=float, default=0.10)
    ap.add_argument("--site-card-wt-mismatch-min-checked", type=int, default=20)
    ap.add_argument("--allow-site-card-wt-mismatch", action="store_true", default=False)
    ap.add_argument("--dedupe-scope", choices=["none", "panel", "all"], default="panel")
    ap.add_argument("--dedupe-lookback-rounds", type=int, default=2)
    ap.add_argument("--proposal-seed", type=int, default=13)
    ap.add_argument("--ensemble-models", type=int, default=5)
    ap.add_argument("--ensemble-max-iter", type=int, default=700)
    ap.add_argument("--min-train-samples", type=int, default=80)
    ap.add_argument("--ehvi-mc", type=int, default=32)
    ap.add_argument("--min-function", type=float, default=0.40)
    ap.add_argument("--min-binding", type=float, default=0.35)
    ap.add_argument("--min-stability", type=float, default=0.40)
    ap.add_argument("--min-plausibility", type=float, default=0.40)
    ap.add_argument("--functional-site-binding-floor", type=float, default=0.15)
    ap.add_argument("--enable-functional-binding-challenger", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--binding-challenger-frac", type=float, default=0.20)
    ap.add_argument("--binding-challenger-min", type=int, default=2)
    ap.add_argument("--binding-challenger-max", type=int, default=24)
    ap.add_argument("--binding-challenger-min-binding", type=float, default=0.02)
    ap.add_argument("--binding-challenger-uncertainty-min", type=float, default=0.08)
    ap.add_argument("--binding-challenger-max-signal", type=float, default=0.35)
    ap.add_argument("--binding-challenger-min-func", type=float, default=0.20)
    ap.add_argument("--min-binding-challenger-selected", type=int, default=3)
    ap.add_argument("--tau-func-green", type=float, default=0.70)
    ap.add_argument("--tau-func-amber", type=float, default=0.45)
    ap.add_argument("--max-per-position", type=int, default=4)
    ap.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-proposal-total", type=int, default=2400)
    ap.add_argument("--diversity-floor", type=float, default=0.20)
    ap.add_argument("--voi-cost-threshold", type=float, default=0.0025)
    ap.add_argument("--voi-patience", type=int, default=2)
    ap.add_argument("--min-iterations", type=int, default=3)
    ap.add_argument("--objective-improvement-eps", type=float, default=0.005)
    ap.add_argument("--min-budget-fraction-before-voi-stop", type=float, default=0.75)
    ap.add_argument("--force-regenerate-proposals", action="store_true", default=False)
    ap.add_argument("--model-weights-dir", default=None)
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--cpu-embeddings", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gpu-embeddings", dest="cpu_embeddings", action="store_false")
    ap.add_argument("--with-janus-final", action="store_true", default=False)
    ap.add_argument("--janus-cmd", default=None)
    ap.add_argument("--janus-repo", default=None)
    ap.add_argument("--fast-binding-check", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gnina-bin", default=None)
    ap.add_argument("--binding-ld-library-path", default=None)
    ap.add_argument("--binding-cnn-model", default="fast")
    ap.add_argument("--binding-workers", type=int, default=4)
    ap.add_argument("--binding-cpu-per-job", type=int, default=1)
    ap.add_argument("--binding-autobox-add", type=float, default=6.0)
    ap.add_argument("--binding-progress-every", type=int, default=10)
    ap.add_argument("--binding-relax-mutants", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--binding-relax-max-iterations", type=int, default=120)
    ap.add_argument("--binding-relax-heavy-restraint-k", type=float, default=25.0)
    ap.add_argument("--binding-max-variants", type=int, default=0)
    ap.add_argument("--binding-score-all", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(os.environ.get("OUTDIR", "./data"))
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OUTDIR"] = str(outdir)

    if not args.skip_input_prep:
        prep_cmd = [sys.executable, "scripts/swarm/14a_prepare_inputs.py", "--outdir", str(outdir)]
        forward_flags = {
            "--input-spec": args.input_spec,
            "--protein-source": args.protein_source,
            "--protein-path": args.protein_path,
            "--protein-id": args.protein_id,
            "--protein-chain": args.protein_chain,
            "--fasta-source": args.fasta_source,
            "--fasta-path": args.fasta_path,
            "--fasta-accession": args.fasta_accession,
            "--ligand-source": args.ligand_source,
            "--ligand-path": args.ligand_path,
            "--ligand-id": args.ligand_id,
            "--ligand-smiles": args.ligand_smiles,
        }
        for k, v in forward_flags.items():
            if v:
                prep_cmd += [k, v]
        run(prep_cmd, env=env)

    # ---- API pack ----
    api_outdir = outdir / "swarm_api"
    api_ctx = api_outdir / "context_api.json"

    should_run_api = False
    if not args.skip_api:
        if args.force_api:
            should_run_api = True
        elif not api_ctx.exists():
            should_run_api = True

    if should_run_api:
        cmd = [sys.executable, "-m", "scripts.swarm.api.build_context_pack"]
        accession = args.accession
        if not accession:
            fasta_path = resolve_canonical_fasta(outdir)
            accession = infer_uniprot_accession(outdir, fasta_path=fasta_path)

        if accession:
            cmd += ["--accession", accession]
        elif args.protein_name:
            cmd += ["--protein-name", args.protein_name]
            if args.organism_id:
                cmd += ["--organism-id", args.organism_id]
            if args.organism_name:
                cmd += ["--organism-name", args.organism_name]
            if args.reviewed_only:
                cmd += ["--reviewed-only"]
        else:
            print("[swarm] Skipping API pack: no accession or protein-name provided.")
            should_run_api = False

        if should_run_api:
            cmd += ["--outdir", str(outdir)]
            if args.offline:
                cmd += ["--offline"]
            if args.include_chembl:
                cmd += ["--include-chembl"]
            if args.include_hmmer:
                cmd += ["--include-hmmer"]
            if args.include_mcsa:
                cmd += ["--include-mcsa"]
            if args.chembl_query:
                cmd += ["--chembl-query", args.chembl_query]
            if args.chembl_organism:
                cmd += ["--chembl-organism", args.chembl_organism]
            if args.ligand_name:
                cmd += ["--ligand-name", args.ligand_name]
            if args.ligand_cid:
                cmd += ["--ligand-cid", args.ligand_cid]

            run(cmd, env=env)

    # ---- Local packs ----
    run([sys.executable, "scripts/swarm/15a_build_context_pack.py", "--outdir", str(outdir)], env=env)
    run([sys.executable, "scripts/swarm/15b_build_site_cards.py", "--outdir", str(outdir)], env=env)

    if not args.skip_prolif:
        try:
            run(
                [
                    sys.executable,
                    "scripts/swarm/15f_enrich_site_cards_prolif.py",
                    "--outdir",
                    str(outdir),
                    "--max-poses",
                    str(max(1, args.prolif_max_poses)),
                ],
                env=env,
            )
        except subprocess.CalledProcessError:
            print("[swarm] Warning: ProLIF enrichment failed; continuing with base site cards.")

    if args.evmutation_priors or args.eve_priors or args.deepsequence_priors:
        cmd = [sys.executable, "scripts/swarm/15g_build_sequence_priors.py", "--outdir", str(outdir)]
        if args.evmutation_priors:
            cmd += ["--evmutation", args.evmutation_priors]
        if args.eve_priors:
            cmd += ["--eve", args.eve_priors]
        if args.deepsequence_priors:
            cmd += ["--deepsequence", args.deepsequence_priors]
        run(cmd, env=env)

    print("[swarm] Done. Outputs:")
    print("  -", outdir / "swarm" / "context_pack.json")
    print("  -", outdir / "swarm" / "site_cards.jsonl")

    if bool(args.run_recursive):
        recursive_cmd = [
            sys.executable,
            "scripts/swarm/20_run_recursive_adaptive_flow.py",
            "--outdir", str(outdir),
            "--start-round", str(int(args.start_round)),
            "--iterations", str(max(1, int(args.iterations))),
            "--panel-total", str(int(args.panel_total)),
            "--global-panel-budget", str(max(0, int(args.global_panel_budget))),
            "--final-max-candidates", str(max(0, int(args.final_max_candidates))),
            "--proposal-total", str(int(args.proposal_total)),
            "--proposal-max-per-position", str(int(args.proposal_max_per_position)),
            "--max-mutations-per-variant", str(max(1, int(args.max_mutations_per_variant))),
            "--multi-point-fraction", str(float(args.multi_point_fraction)),
            "--multi-seed-size", str(max(8, int(args.multi_seed_size))),
            "--multi-max-candidates", str(max(0, int(args.multi_max_candidates))),
            "--multi-min-position-separation", str(max(0, int(args.multi_min_position_separation))),
            "--multi-max-position-separation", str(max(0, int(args.multi_max_position_separation))),
            "--critical-blosum-min", str(int(args.critical_blosum_min)),
            "--strict-evo-conservation-threshold", str(float(args.strict_evo_conservation_threshold)),
            "--strict-evo-blosum-min", str(int(args.strict_evo_blosum_min)),
            "--functional-exploratory-blosum-min", str(int(args.functional_exploratory_blosum_min)),
            "--functional-exploratory-max-extra", str(max(0, int(args.functional_exploratory_max_extra))),
            "--functional-exploratory-ligand-shell", str(float(max(0.0, args.functional_exploratory_ligand_shell))),
            "--site-card-wt-mismatch-max-frac", str(float(args.site_card_wt_mismatch_max_frac)),
            "--site-card-wt-mismatch-min-checked", str(max(1, int(args.site_card_wt_mismatch_min_checked))),
            "--dedupe-scope", str(args.dedupe_scope),
            "--dedupe-lookback-rounds", str(max(0, int(args.dedupe_lookback_rounds))),
            "--proposal-seed", str(int(args.proposal_seed)),
            "--ensemble-models", str(int(args.ensemble_models)),
            "--ensemble-max-iter", str(int(args.ensemble_max_iter)),
            "--min-train-samples", str(int(args.min_train_samples)),
            "--ehvi-mc", str(int(args.ehvi_mc)),
            "--min-function", str(float(args.min_function)),
            "--min-binding", str(float(args.min_binding)),
            "--min-stability", str(float(args.min_stability)),
            "--min-plausibility", str(float(args.min_plausibility)),
            "--functional-site-binding-floor", str(float(args.functional_site_binding_floor)),
            "--binding-challenger-frac", str(float(args.binding_challenger_frac)),
            "--binding-challenger-min", str(max(0, int(args.binding_challenger_min))),
            "--binding-challenger-max", str(max(0, int(args.binding_challenger_max))),
            "--binding-challenger-min-binding", str(float(args.binding_challenger_min_binding)),
            "--binding-challenger-uncertainty-min", str(float(max(0.0, args.binding_challenger_uncertainty_min))),
            "--binding-challenger-max-signal", str(float(max(0.0, args.binding_challenger_max_signal))),
            "--binding-challenger-min-func", str(float(args.binding_challenger_min_func)),
            "--min-binding-challenger-selected", str(max(0, int(args.min_binding_challenger_selected))),
            "--tau-func-green", str(float(args.tau_func_green)),
            "--tau-func-amber", str(float(args.tau_func_amber)),
            "--max-per-position", str(int(args.max_per_position)),
            "--max-proposal-total", str(max(120, int(args.max_proposal_total))),
            "--diversity-floor", str(float(args.diversity_floor)),
            "--voi-cost-threshold", str(float(max(0.0, args.voi_cost_threshold))),
            "--voi-patience", str(int(max(1, args.voi_patience))),
            "--min-iterations", str(max(1, int(args.min_iterations))),
            "--objective-improvement-eps", str(float(max(0.0, args.objective_improvement_eps))),
            "--min-budget-fraction-before-voi-stop", str(float(args.min_budget_fraction_before_voi_stop)),
            "--binding-cnn-model", str(args.binding_cnn_model),
            "--binding-workers", str(max(1, int(args.binding_workers))),
            "--binding-cpu-per-job", str(max(1, int(args.binding_cpu_per_job))),
            "--binding-autobox-add", str(float(args.binding_autobox_add)),
            "--binding-progress-every", str(max(1, int(args.binding_progress_every))),
            "--binding-relax-max-iterations", str(max(1, int(args.binding_relax_max_iterations))),
            "--binding-relax-heavy-restraint-k", str(float(args.binding_relax_heavy_restraint_k)),
        ]
        if args.focus_round is not None:
            recursive_cmd += ["--focus-round", str(int(args.focus_round))]
        if bool(args.force_regenerate_proposals):
            recursive_cmd += ["--force-regenerate-proposals"]
        if bool(args.functional_site_hard_filter):
            recursive_cmd += ["--functional-site-hard-filter"]
        if bool(args.near_functional_hard_filter):
            recursive_cmd += ["--near-functional-hard-filter"]
        if bool(args.functional_exploratory_enable):
            recursive_cmd += ["--functional-exploratory-enable"]
        else:
            recursive_cmd += ["--no-functional-exploratory-enable"]
        if bool(args.allow_site_card_wt_mismatch):
            recursive_cmd += ["--allow-site-card-wt-mismatch"]
        if args.model_weights_dir:
            recursive_cmd += ["--model-weights-dir", str(args.model_weights_dir)]
        if args.hf_home:
            recursive_cmd += ["--hf-home", str(args.hf_home)]
        if bool(args.cpu_embeddings):
            recursive_cmd += ["--cpu-embeddings"]
        else:
            recursive_cmd += ["--gpu-embeddings"]
        if bool(args.adaptive):
            recursive_cmd += ["--adaptive"]
        else:
            recursive_cmd += ["--no-adaptive"]
        if bool(args.fast_binding_check):
            recursive_cmd += ["--fast-binding-check"]
        else:
            recursive_cmd += ["--no-fast-binding-check"]
        if bool(args.enable_functional_binding_challenger):
            recursive_cmd += ["--enable-functional-binding-challenger"]
        else:
            recursive_cmd += ["--no-enable-functional-binding-challenger"]
        if bool(args.binding_relax_mutants):
            recursive_cmd += ["--binding-relax-mutants"]
        else:
            recursive_cmd += ["--no-binding-relax-mutants"]
        if args.gnina_bin:
            recursive_cmd += ["--gnina-bin", str(args.gnina_bin)]
        if args.binding_ld_library_path:
            recursive_cmd += ["--binding-ld-library-path", str(args.binding_ld_library_path)]
        if int(args.binding_max_variants) > 0:
            recursive_cmd += ["--binding-max-variants", str(int(args.binding_max_variants))]
        if bool(args.binding_score_all):
            recursive_cmd += ["--binding-score-all"]
        else:
            recursive_cmd += ["--no-binding-score-all"]
        if bool(args.with_janus_final):
            recursive_cmd += ["--with-janus-final"]
        if args.janus_cmd:
            recursive_cmd += ["--janus-cmd", str(args.janus_cmd)]
        if args.janus_repo:
            recursive_cmd += ["--janus-repo", str(args.janus_repo)]
        run(recursive_cmd, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
