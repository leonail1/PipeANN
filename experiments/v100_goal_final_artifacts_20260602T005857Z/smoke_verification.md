# V100 Final Artifact Smoke Verification

- Scope: final generator/PPT wording update for `experiments/v100_goal_final_artifacts_20260602T005857Z`.
- Environment note: `PYTHONNOUSERSITE=1` is required for scripts that import Matplotlib on v100 because the user-site NumPy 2.2.6 conflicts with the system Matplotlib build; with `PYTHONNOUSERSITE=1`, NumPy/Matplotlib import as 1.21.5/3.5.1.

## Checks

- `python3 -m py_compile scripts/generate_v100_goal_artifacts.py scripts/generate_v100_ppt_refresh.py`: PASS.
- Current artifact data consistency: PASS.
  - selected rows: 200.
  - targeted profile rows: 200.
  - PQ strategy rows: 100.
  - selected/profile route, `search_l`, and `beamwidth` mismatches: 0.
  - selected beamwidth distribution: `{4: 165, 8: 35}`.
  - PQ rows missing `v3_beamwidth`: 0.
  - final ARIS review dynamic wording includes `selected serving config route/L/beamwidth`.
- PPT refresh smoke with `PYTHONNOUSERSITE=1`: PASS.
  - temporary output wrote 21 artifacts.
  - temporary `dynamic-update.tex` matched the current checked tree after idempotent refresh.
- Goal-artifact generator smoke with `PYTHONNOUSERSITE=1`: PASS.
  - temporary output directory: `experiments/v100_goal_artifacts_smoke_beamwidth_20260602Ttmp`.
  - selected rows: 200.
  - generated figures: 3.
  - selected beamwidth distribution: `{4: 165, 8: 35}`.
  - selected/profile route, `search_l`, and `beamwidth` mismatches: 0.
  - final ARIS dynamic wording check: PASS.
  - temporary smoke directories were removed after verification.

## Provenance

- Reviewed scripts: `scripts/generate_v100_goal_artifacts.py`, `scripts/generate_v100_ppt_refresh.py`.
- Final evidence directory: `experiments/v100_goal_final_artifacts_20260602T005857Z`.
- PPT refresh evidence directory: `experiments/v100_ppt_refresh_20260602T011651Z`.
