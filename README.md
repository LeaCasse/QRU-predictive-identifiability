# QRU predictive identifiability

Code and data for *Hidden spectral paths and predictive identifiability in quantum data reuploading*, by Léa Cassé, Bernhard Pfahringer and Albert Bifet.

## Run

Python 3.12.

```bash
python -m pip install -r requirements.txt
python reproduce.py checks
python reproduce.py figures
```

Generated results go to `build/runs/`, figures to `build/figures/`. Reference results are in `results/`.

## Experiments

```bash
python reproduce.py <target>
python reproduce.py <target> --help
```

| Target | Calculation |
|---|---|
| `twins` | Present-equivalent circuits and drift responses |
| `path-geometry` | Two-layer amplitudes and known-dephasing constructions |
| `attainable-set` | Complete two-layer constraints and circuit reconstruction |
| `phase-design` | Layer contrasts and phase tomography |
| `control-design` | Layer, input and phase selection with shot bounds |
| `depth-family` | Adaptive discrimination at increasing depth |
| `streams` | 32 paired streams, four shots per step, horizons one and five |
| `branch-selection` | 256 paired starts across five training modes |
| `branch-robustness` | Two additional training targets, 128 paired starts each |
| `hardware-noise` | Dephasing fit and held-out checks on seed-203 counts |
| `metric-noise` | Gate-error models constrained by archived SX metrics |
| `real` | Chronological Sunspots/Nile replay |

`depth-family` assumes known present amplitude and phase. Training results retain raw circuits and clean least-squares completions separately.

Use `python reproduce.py --help` for all targets. Experiment options, including seeds and output paths, are listed by each target's `--help`.

## Files

| Path | Contents |
|---|---|
| `qru/` | Circuit models, path geometry, inference and controls |
| `experiments/` | Experiment entry points |
| `results/` | Numerical results and raw hardware counts |
| `data/` | Series, figure data, source hashes and provenance |
| `figures/` | Figure scripts and 24 article PDFs |
| `tests/` | Circuit identities, risks, protocols and data checks |
| `hardware/` | IBM acquisition and offline noise analysis |

## Symbolic constraints

```bash
python -m pip install -r requirements-symbolic.txt
python reproduce.py algebraic-fiber
```

Exports the real polynomial constraints for the depth-three fiber at `(1, 2, 4)`. Installing SymPy also enables the symbolic checks in `python reproduce.py checks`.

## Hardware

Offline diagnostics use the included counts and require no IBM account. A calibrated replay requires an independent historical calibration snapshot and the executed bound circuits:

```bash
python hardware/calibrated_replay.py --help
```

The seed-203 archives contain mean SX/readout errors but lack T1/T2, gate durations, asymmetric readout probabilities and bound ISA circuits.

For live acquisition:

```bash
python -m pip install -r hardware/requirements.txt
python hardware/probe.py --help
```

Live acquisition requires IBM Quantum credentials.
