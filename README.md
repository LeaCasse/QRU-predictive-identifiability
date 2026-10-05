# Hidden spectral paths and predictive identifiability in quantum data reuploading

Research code and results for the article by Léa Cassé, Bernhard Pfahringer and Albert Bifet. The question is whether identical present responses determine future responses when input encoding changes, and when a measurement on the target resolves the ambiguity.

## Run

```bash
python -m pip install -r requirements.txt
python reproduce.py checks
python reproduce.py figures
```

## Contents

| Directory | Contents |
|---|---|
| `qru/` | Circuit simulator, spectral paths, derivatives, information, adaptation and classical controls |
| `experiments/` | Seeded experiments with explicit protocols and JSON outputs |
| `data/` | Sunspots/Nile observations, publication plot data, source hashes and table provenance |
| `results/` | Frozen reference tables, pilot trials, 32-seed controlled streams and hardware records |
| `figures/` | Scripts and vector PDFs for all 21 article/appendix figures |
| `hardware/` | Optional IBM experiment runner; archived-count analysis requires no account |
| `tests/` | Independent circuit checks, path identities, shot budgets, bootstrap and hardware gates |

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

`streams` uses all 32 paired seeds (1300-1331), ten scenarios and the real-data replay. `real` runs only the chronological Sunspots/Nile forecasts. Each trial uses three query policies and horizons 1 and 5. The figures can be rebuilt directly from the included results.
