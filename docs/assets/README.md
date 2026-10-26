# FAC-Det

**Fine-grained appearance calibration for traffic-sign detection with limited annotations**

FAC-Det combines local appearance evidence, shared attribute matching, candidate verification and quality calibration for fine-grained traffic-sign detection.

## Source

| Component | Location |
|---|---|
| Detector and FACB | `models/GroundingDINO/` |
| Attribute representations | `datasets/appearance_prompt.py` |
| Quality calibration | `util/facb_fixed_quality_online.py` |
| Model and runtime utilities | `util/`, `groundingdino/` |

## Dependencies

Python 3.12 · PyTorch 2.6 · CUDA 12.4

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -r requirements-build.txt
python -m pip install --no-build-isolation ./models/GroundingDINO/ops
```

Building the operator requires CUDA Toolkit 12.4 and a compatible C++ compiler. Windows/Python 3.12 dependency constraints are provided in `constraints-win-py312.txt`.

## Research support

This repository provides the method implementation. Experiment configurations, data splits, training records, version histories and model weights are not included in this public release. Please contact the authors for reproduction support.

## Acknowledgments

FAC-Det is developed on top of [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino), the third-party implementation of Grounding DINO by Zuwei Long and Wei Li. We thank its authors and contributors for making their implementation publicly available, and acknowledge the original Grounding DINO authors and the upstream projects used in this codebase.

The original upstream copyright and license notices are retained. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
