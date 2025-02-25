# Minimal Spin implementation

## Reproduce the ABX results

- Install this package
  - For example with uv: `uv sync` or `uv sync --group example`. And activate the environment
- `cd examples`
- Modify "template.yaml" to add the paths to the manifests of train-clean-100, dev-clean and dev-other
  - Manifests should be CSV files with columns: "fileid", "path", "num_frames", "speaker"
- Train HuBERT + spin on a machine with a single GPU: `python hubert.py template.yaml`. It should take less than 30 min.
- Use the "abx.py" script to extract features and compute the scores.
