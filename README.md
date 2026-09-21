# BSc Intro Data Science - CSD4 Baseline Prediction

- Course: BSc Computer Science.

## Contents

Image-classification baseline coursework using feature extraction, logistic regression, and evaluation metrics.

## Files

Template or reference material:

- `assignment/CSD4reference.ipynb`

My solution notebooks:

- `solutions/CSD4.ipynb`

My submitted answers:

- `results/csd4_answers.csv`

## Tech Stack

- Python notebooks.
- Main Python packages: matplotlib, numpy, pandas, requests, scikit-image, scikit-learn, notebook.
- Jupyter-compatible local review flow.

## Notes

- The image dataset is supplied separately. The runnable notebook uses a local dataset directory.

## Run

Supply the extracted `ebay_boys_girls_shirts` course dataset:

```sh
uv run --python 3.11 python scripts/run_notebook.py /path/to/ebay_boys_girls_shirts
```

This runs every solution code cell using 64 training images and 32 test images per class. It fits and evaluates all three logistic-regression models, checks probabilities and confusion matrices, renders plots, and exports calculated answers.

Add `--full` to use 2,000 training images per class and 500 test images per class. Use `--output-dir PATH` to keep generated plots and CSVs; otherwise they are temporary. Submitted files are never overwritten. For interactive use, set `SHIRTS_DATASET` before opening the solution notebook.
