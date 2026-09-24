# UbiStruct-QA

UbiStruct-QA scores candidate protein assemblies using agreement between model
groups, interface geometry, predictor confidence and structural references.
It includes the CASP17 configurations for immune complexes H2441 and H2444,
peptide assemblies T2463 and T2464, and the T2461 protein cage. Group `ubi`
submitted quality estimates for all five targets.

Scores rank models within each target pool. They are not probabilities or
measurements of accuracy against an experimental target structure. The code
runs locally without an account, API token or online service.

## Install

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Try the example

```bash
python examples/offline_demo.py --output-dir demo-output
```

The example creates 54 synthetic models, runs all three scorers and checks the
five output files. It tests the workflow, not scientific accuracy. It requires
no network access or competition data.

## Score models

Keep each CASP model filename unchanged. Place the extracted models under
`data/raw/casp/H2441`, `H2444`, `T2463o`, `T2464o` and `T2461o`. Put the MassiveFold archives in `incoming/`.

```bash
ubistruct-immune run-all \
  --h2441-casp data/raw/casp/H2441 \
  --h2441-mf incoming/H2441_all_pdbs_MassiveFold.tar.gz \
  --h2444-casp data/raw/casp/H2444 \
  --h2444-mf incoming/H2444_all_pdbs_MassiveFold.tar.gz \
  --author YOUR-REGISTRATION-CODE

python scripts/fetch_public_templates.py
ubistruct-fibril --author YOUR-REGISTRATION-CODE
ubistruct-cage --author YOUR-REGISTRATION-CODE
```

Use each command's `--help` for input and output paths. Each scorer writes
a score table, diagnostics and a CASP-format QA file. The immune and peptide
scorers also save feature files. The local
validators check coverage, duplicate identifiers and score bounds. They do not
confirm acceptance by the CASP submission server.

The template downloader obtains 7QV6 and the biological assembly of 6FDB from
RCSB. Run it once before the peptide or cage workflow, or provide those files
with the corresponding path options. The download requires network access;
scoring uses the files locally. Downloader options:

```bash
python scripts/fetch_public_templates.py --help
```

The 7QV6 comparison has zero weight in the T2464 score. The 6FDB comparison is
part of the T2461 scorer. Cα distance pairs from 4RL2, 6SVK and
ImmuneBuilder component predictions are included as constants in `src/`.
Prediction coordinate archives, generated results and external software are
not included.

## Methods and limits

See [methodology](docs/methodology.md) for score weights, source grouping,
confidence handling and limitations, and [references](docs/references.md) for
the scientific methods. The scripts in `scripts/` test how rankings change
when sources or score weights change.

Grouping models reduces the influence of unequal sampling, but related
predictors can still share errors. Some confidence fields use fallbacks when
predictor metadata are absent. Reference-distance comparisons use available
Cα pairs and skip missing pairs; they are not the original all-atom lDDT.
Experimental target structures are needed to measure predictive accuracy.

## Development

```bash
python -m pip install '.[dev]'
python -m unittest discover -s tests -v
ruff check src scripts tests examples
python -m build --no-isolation
```

Licensed under Apache-2.0. See `CITATION.cff` for software citation information.
