# OpenHarmony ANNS Acceptance Adapter

This directory contains the PipeANN-side command manifest, configs, and
lightweight evidence for the independent `OpenHarmony-ANNS-Test` repository.
The tests are not modified from PipeANN; they call the five public commands
through `adapter_manifest.pipeann.yaml`.

Smoke command on v100:

```bash
cd /mnt/nvme1n1/OpenHarmony-ANNS-Test
source .venv/bin/activate
python -m pytest \
  --config /mnt/nvme1n1/PipeANN-github/acceptance/sift_smoke_config.yaml -q
```

The current adapter backend uses exact filtered search for small candidate sets
and a FAISS IVFFlat + exact-rerank route for high-selectivity workloads. It
emits trace fields such as candidate count and backend route without changing
the external acceptance-test interface.

Final v100 evidence is summarized in `FINAL_ACCEPTANCE_EVIDENCE.md`. Generated
work directories and runtime logs are intentionally ignored; commit only the
configs, adapter, and lightweight result artifacts.
