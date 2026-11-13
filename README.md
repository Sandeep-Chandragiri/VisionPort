# VisionPort

A GPU-trained, GPU-served vision system for Indian traffic.

## Current version

v0.1 - pretrained YOLOv8n demo, FastAPI gateway, request trace

## Codebase tour

| Part | Location | Will contain |
|---|---|---|
| Data pipeline | `data/`, `train/dataset.py` | Dataset loading and preprocessing for training |
| Training loop | `train/train.py` | Model training and optimisation loop |
| Architecture file | `train/model.py` | Model architecture and configuration |
| Evaluation harness | `eval/` | Model evaluation and performance metrics |
| Optimisation script | `optimise/` | Model optimisation for efficient inference |
| Triton config | `serve/triton/` | NVIDIA Triton model-serving configuration |

## Environment

See `env_report.md`. Recreate with:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## How to run

```powershell
uvicorn serve.gateway_trace:app --reload
```

Then open `http://127.0.0.1:8000/docs` in your browser.

## Known limitations

- Pretrained on COCO, not on Indian traffic.
- No auto-rickshaw class.
