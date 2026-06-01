# Transformers Workspace

This workspace contains small experiments and utility scripts for working with Hugging Face and transformer-based models. It is intended as a lightweight development area for testing ideas, building custom pipelines, and iterating on local Python code.

## Repository Structure

- `hugging_face1/curtom_pipeline.py` - custom pipeline-related code and experiments
- `hugging_face1/pepe1.py` - additional Python script for local experimentation

## Environment

The workspace is designed to be used with a local Python virtual environment in `.venv/`. If you are working in VS Code, select the interpreter from that environment before running scripts or notebooks.

## Working With Data

Large datasets and generated artifacts should stay out of version control. The root `.gitignore` excludes common local environment and editor folders, along with typical dataset and ML artifact formats such as CSV, JSONL, Parquet, NumPy arrays, pickle files, and model checkpoints.

## Suggested Workflow

1. Create or activate the `.venv` environment.
2. Install the Python dependencies required for the script you want to run.
3. Open the relevant file in `hugging_face1/` and execute it from the configured interpreter.
4. Keep datasets, model outputs, and temporary experiments in ignored local folders.

## Notes

- `.vscode/` remains untracked so local editor settings stay personal to each machine.
- If you add new scripts or notebooks, document their purpose here so the workspace stays easy to navigate.
