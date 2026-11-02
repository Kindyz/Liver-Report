# Liver CT report generation

- `train/train_liver_qwen.py`: training entry point
- `config/260712_liver_qwen.yaml`: training configuration
- `data/Dataset/Liver_dataset.py`: dataset loader
- `Models/liver_adapter_model.py`: image-to-language adapter
- `data/data.xlsx`: tabular CT image paths and report data
- `preprocess/get_emb.py`: Merlin image embedding extraction
- `util/train_utils.py`: training utilities

## Project-relative paths

Run scripts from any working directory. Their default paths are resolved from the project root.
Place the Qwen model in `models/qwen-7b/`, the Merlin weights in `models/merlin/weights/`,
the split file in `splits/split_8_1_1.json`, and CT images under `data/images/` when the workbook is migrated to relative paths.
Preprocessing writes `embedding/liver_embeddings.h5`; training writes to `outputs/liver/`.
Supply local files at those locations before running the pipeline.

Training and preprocessing require a CUDA-enabled PyTorch installation and an NVIDIA GPU.
Use `CUDA_VISIBLE_DEVICES` to select a GPU; preprocessing uses the first visible GPU,
and Accelerate assigns a GPU to each training process. Training uses bf16 when the GPU
supports it and fp16 otherwise. Set `runtime.use_tf32` in the YAML configuration to
enable TF32 matrix multiplication on supported GPUs.

The existing `data/data.xlsx` is excluded from this cleanup. It still contains original
image paths and Chinese clinical reports; preprocessing continues to read its image
paths as recorded. The duplicate-ID utility expects input under
`data/registered_liver_ct/` and writes copies to `data/duplicate_liver_ct/`.
