# Hidden spectral paths and predictive identifiability in quantum data reuploading

Research code and results for the article by Léa Cassé, Bernhard Pfahringer and Albert Bifet. The study identifies attainable paths with the same present response and different controlled futures. It characterizes the complete two-layer sinusoidal path fiber, computes its exact prediction risk and tests target measurements at a fixed shot budget. An exact construction extends the mechanism to known interlayer dephasing.

## Run

```bash
python -m pip install -r requirements.txt
python reproduce.py checks
python reproduce.py path-geometry
python reproduce.py figures
```

## Contents

| Directory | Contents |
|---|---|
| `qru/` | Circuit and density-matrix simulators, attainable amplitudes, information, adaptation and classical controls |
| `experiments/` | Seeded experiments with explicit protocols and JSON outputs |
| `data/` | Sunspots/Nile observations, publication plot data, source hashes and table provenance |
| `results/` | Reference tables, controlled streams, exact fiber/dephasing calculations and raw hardware records |
| `figures/` | Scripts and vector PDFs for all 22 main and supporting figures |
| `hardware/` | Optional IBM acquisition runner |
| `tests/` | Independent circuit identities, minimax risks, shot budgets, bootstrap and hardware gates |

## Experiments

```bash
python reproduce.py twins
python reproduce.py streams
python reproduce.py real
python reproduce.py observability
python reproduce.py two-qubit
python reproduce.py optimizer
python reproduce.py detectability
python reproduce.py finite-alternatives
```

Calculations write to `build/`. Included results and figures can be used directly.

`path-geometry` checks explicit two-layer amplitudes on 10,000 circuits and the dephasing construction on eleven channel settings. The complete-fiber and minimax statements are proved in the article.

`streams` uses 32 paired seeds (1300 through 1331), ten scenarios, three query policies and horizons one and five. Each controlled stream uses four target shots per step. `real` runs the chronological Sunspots/Nile replay. `figures` builds vector PDFs and 300 dpi PNGs from the included data.

The density-matrix model applies a specified common channel between layers. Its channel-dependent preparation assumes known dephasing. Hardware records retain the staged baseline-equivalence test and require no account to analyze. Live acquisition through `hardware/probe.py` uses a separate IBM account.
