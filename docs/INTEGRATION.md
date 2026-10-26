# Integration

The model implementation is in `models/GroundingDINO/groundingdino.py`. Attribute representations are in `datasets/appearance_prompt.py`; quality-scoring modules are in `util/`.

The generic `main.py` / `engine.py` interfaces require caller-supplied configuration, dataset manifests, class descriptions and compatible weights. These experiment-specific inputs are not included. Contact the authors for reproduction support.
