"""Execute every solution code cell and check computed numerical results."""

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "dataset", type=Path, help="Extracted ebay_boys_girls_shirts directory"
)
parser.add_argument("--full", action="store_true", help="Use original sample sizes")
parser.add_argument("--samples-per-class", type=int, default=64)
parser.add_argument(
    "--output-dir", type=Path, help="New or empty directory for plots and CSVs"
)
args = parser.parse_args()
if args.samples_per_class < 16:
    parser.error("Use at least 16 training images per class")
dataset = args.dataset.expanduser().resolve()
for filename in (
    "boys_train.csv",
    "girls_train.csv",
    "boys_test.csv",
    "girls_test.csv",
):
    if not (dataset / filename).is_file():
        parser.error("Missing dataset file: " + filename)
os.environ["SHIRTS_DATASET"] = str(dataset)
root = Path(__file__).resolve().parents[1]
notebook = root / "solutions" / "CSD4.ipynb"
cells = json.loads(notebook.read_text())["cells"]


class SampleSize(ast.NodeTransformer):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "n_train":
                node.value = ast.Constant(args.samples_per_class)
            elif node.targets[0].id == "n_test":
                node.value = ast.Constant(min(32, args.samples_per_class))
        return node


def check_model(namespace, features, labels):
    import numpy as np
    from sklearn.metrics import confusion_matrix

    model = namespace["mod"]
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(labels), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    assert set(predicted).issubset({0, 1})
    confusion = confusion_matrix(labels, predicted, labels=[0, 1])
    assert confusion.sum() == len(labels)
    np.testing.assert_allclose(
        model.score(features, labels), np.mean(predicted == labels)
    )


def execute(output):
    import numpy as np

    namespace = {"__name__": "__main__"}
    completed = 0
    figures = 0
    original_cwd = Path.cwd()
    try:
        os.chdir(output)
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(
                line for line in cell["source"] if not line.lstrip().startswith("%")
            )
            if not source.strip():
                continue
            tree = ast.parse(source)
            if not args.full:
                tree = SampleSize().visit(tree)
            ast.fix_missing_locations(tree)
            print(f"Executing cell {index}", flush=True)
            exec(compile(tree, f"{notebook.name}:cell{index}", "exec"), namespace)
            completed += 1
            if index == 20:
                check_model(namespace, namespace["x_test_av"], namespace["y_test"])
            elif index == 44:
                check_model(
                    namespace, namespace["x_test_av_channels"], namespace["y_test"]
                )
            elif index == 48:
                check_model(namespace, namespace["x_test"], namespace["y_test"])
            if "plt" in namespace:
                plt = namespace["plt"]
                for figure_number in plt.get_fignums():
                    plt.figure(figure_number).savefig(
                        f"cell-{index}-figure-{figure_number}.png"
                    )
                    figures += 1
                plt.close("all")
        assert np.isfinite(namespace["x_train"]).all()
        conf = namespace["pixel_conf"]
        assert conf.shape == (2, 2) and conf.sum() == len(namespace["y_test"])
        answers = namespace["ans"]
        np.testing.assert_allclose(answers["Q6"], conf[1, 1] / conf[1].sum())
        np.testing.assert_allclose(answers["Q7"], conf[0, 0] / conf[:, 0].sum())
        np.testing.assert_allclose(answers["Q8"], conf[1, 1] / conf[:, 1].sum())
        for question in ("Q1", "Q2", "Q3", "Q5", "Q6", "Q7", "Q8", "Q9"):
            assert np.isfinite(answers[question]) and 0 <= answers[question] <= 1
        assert len(list(Path(".").glob("CSD4_*.csv"))) == 1
        print(
            f"PASS: {completed} solution code cells; {figures} plots; real-image computation and numerical checks"
        )
    finally:
        os.chdir(original_cwd)


if args.output_dir:
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    execute(output)
else:
    with tempfile.TemporaryDirectory(prefix="course-notebook-") as directory:
        execute(Path(directory))
