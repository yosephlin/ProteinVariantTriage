# Swarm API Context Pack

This folder contains a minimal, offline-friendly scaffold to build an API-derived context pack for SWARM.

## Outputs
Runs produce these files under `OUTDIR/swarm_api/`:
- `context_api.json`
- `residue_constraints.jsonl`
- `source_cache/` (raw cached API responses)

## Quick Start

```bash
python -m scripts.swarm.api.build_context_pack --accession <UNIPROT_ACCESSION> --outdir /content/run_LB
```

Resolve by name + organism (if you don't have accession):

```bash
python -m scripts.swarm.api.build_context_pack --protein-name "<PROTEIN_NAME>" --organism-id <NCBI_TAXON_ID> --reviewed-only --outdir /content/run_LB
```

Optional ChEMBL ligand priors:

```bash
python -m scripts.swarm.api.build_context_pack --accession <UNIPROT_ACCESSION> --include-chembl --outdir /content/run_LB
```

Optional PubChem SMILES:

```bash
python -m scripts.swarm.api.build_context_pack --accession <UNIPROT_ACCESSION> --ligand-name "<LIGAND_NAME>" --outdir /content/run_LB
```

Optional M-CSA catalytic constraints:

```bash
python -m scripts.swarm.api.build_context_pack --accession <UNIPROT_ACCESSION> --include-mcsa --outdir /content/run_LB
```

Optional HMMER evolutionary priors (homolog-driven allowed substitutions):

```bash
python -m scripts.swarm.api.build_context_pack --accession <UNIPROT_ACCESSION> --include-hmmer --hmmer-max-hits 120 --outdir /content/run_LB
```

Validate outputs:

```bash
python -m scripts.swarm.api.validate_context_pack --outdir /content/run_LB/swarm_api
```

Offline-only (cache required):

```bash
python -m scripts.swarm.api.build_context_pack --accession <UNIPROT_ACCESSION> --outdir /content/run_LB --offline
```

## Notes
- Uses UniProt for curated residue features.
- Uses PDBe ligand sites (best-effort parsing) for observed ligand contact evidence.
- Uses PDBe interface residues and annotations (best-effort parsing) for additional soft constraints.
- Uses InterPro domains (best-effort parsing) for domain labels.
- Uses M-CSA catalytic residue annotations (optional) for stronger do-not-mutate constraints.
- Uses HMMER PHMMER + UniProt sequence pulls (optional) to estimate per-position evolutionary allowed substitutions.
- Caching is deterministic and stored under `source_cache/`.
