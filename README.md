# BSc Intro Data Science - CSD4 Baseline Prediction

- Course: BSc Computer Science.
- Available copy: 2019.
- Supplied exercise material is identified separately below.

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

## Dataset Notes

The original course folders for several data-science assignments contained the large `ebay_boys_girls_shirts` image dataset and tarball. Those files are not tracked in this repository. The recovered notebooks reference the course download URL and recreate the dataset folder when that URL is still available.

## Notes

- The large course image dataset is not tracked; the original notebook downloads the dataset from the course URL.

## Validate

```bash
python3 scripts/check_notebooks.py
```

This check verifies that notebooks parse as JSON and that the removed student identifier does not remain in tracked text files.
