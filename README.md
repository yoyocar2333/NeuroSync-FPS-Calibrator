# NeuroSync — Black-Box FPS Control Calibrator

A portfolio-ready reconstruction of the **system-identification core** documented in the NTU course project *NeuroSync (AimFix)*.

The project estimates hidden FPS control parameters from standardized touch gestures and measured in-game camera rotations:

**screen recording → computer-vision motion tracking → physical velocity → nonlinear least-squares fit → degrees-per-centimeter report**

## What the original course project demonstrated

The course presentation reports:

- OpenCV / NumPy / SciPy / Python 3.11
- 3 standardized gestures × 4 sensitivity configurations = **12 measurements**
- spatial calibration example: **22.6 cm / 2360 px = 0.00958 cm/pixel**
- nonlinear least-squares fit using `scipy.optimize.least_squares`
- reported global fit: **R² = 0.97**
- reported iPad Air 5 fit: `K_sys = 1`, `K_accel = 0.003`
- example physical outputs: **130.8°/cm** low-speed tracking and **145.6°/cm** high-speed flick

Those figures are **reported results from the course presentation**. The original raw videos / 12-point dataset are not included in this reconstruction, so this repository does not claim to independently reproduce R² = 0.97 yet.

## Model

The documented model is

```text
Δθ = K_sys · S · ∫v(t)dt + K_accel · A · ∫v(t)²dt
```

where:

- `Δθ`: measured in-game camera rotation
- `S`: sensitivity setting
- `A`: acceleration-related setting
- `∫v(t)dt`: integrated physical touch velocity
- `∫v(t)²dt`: high-speed / acceleration-sensitive term
- `K_sys`, `K_accel`: hidden engine constants to estimate

This repository now implements both a **configurable reconstruction of the documented CV feature-extraction pipeline** and the nonlinear fitting layer. The CV implementation is an extension reconstructed from the supplied slides, not the original course source code.

## Quick start

```bash
python -m pip install -r requirements.txt

python -m neurosync.extract_video gesture.mp4
python -m neurosync.calibrate data/synthetic_demo.csv
python -m pytest -q
```

`data/synthetic_demo.csv` is a deterministic smoke-test dataset generated from known parameters; it is **not** the original 12-point experiment. `data/example.csv` remains a schema-only template. Replace these with the original measurements before attempting to reproduce the course-project R².

## CSV schema

```text
sensitivity,accel_setting,int_v,int_v2,observed_angle_deg
```

Each row corresponds to one measured gesture/configuration pair.

## Repository structure

```text
.
├── neurosync/
│   ├── __init__.py
│   ├── vision.py           # threshold + connected components + motion integrals
│   ├── extract_video.py    # video → int_v / int_v2 CLI
│   ├── model.py            # nonlinear system-identification model
│   └── calibrate.py        # CSV → fitted constants CLI
├── tests/
│   ├── test_model.py
│   └── test_vision.py
├── data/
│   ├── example.csv
│   └── synthetic_demo.csv
├── docs/
│   └── EVIDENCE.md
├── manifest.json
└── requirements.txt
```

## Agent-skill layer

The course project also proposed wrapping the calibrator as an agent-callable paid skill with a machine-readable manifest and escrow contract.

The presentation distinguishes clearly between:

- **implemented:** calibrator core (CV + physical model + least-squares fitting)
- **designed:** manifest / escrow workflow
- **future work:** testnet deployment, transaction hashes, TEE/ZK-ML verification, multi-skill marketplace

That distinction is preserved here to avoid overstating the blockchain portion.

## Next reproducibility step

To turn this into a fully reviewable research/engineering artifact, add:

1. the original gesture videos;
2. the extracted per-frame centroid/velocity traces;
3. the 12 observed camera-angle measurements;
4. predicted-vs-measured and residual plots on the original measurements;
5. an explicit comparison between the reconstructed tracker and the original implementation, if the original source becomes available.

The repository already reports MAE/RMSE/R² for any supplied calibration CSV and includes a video → `int_v`/`int_v2` reconstruction.

Once those files are available, the reported R² can be independently reproduced from one command.
