from flask import Flask, render_template, request
from sklearn.datasets import load_iris
import numpy as np

app = Flask(__name__)

iris = load_iris()
FEATURES = iris.data
FEATURE_NAMES = [name.replace(" (cm)", "") for name in iris.feature_names]
CLASS_NAMES = iris.target_names.tolist()

OPERATIONS = {
    "add": ("Add", "+"),
    "subtract": ("Subtract", "-"),
    "multiply": ("Multiply", "×"),
    "divide": ("Divide", "÷"),
    "square": ("Square", "x²"),
    "sqrt": ("Square root", "√x"),
}


def transform(values, operation, scalar):
    if operation == "add":
        return values + scalar
    if operation == "subtract":
        return values - scalar
    if operation == "multiply":
        return values * scalar
    if operation == "divide":
        return values / scalar
    if operation == "square":
        return np.square(values)
    return np.sqrt(values)


@app.get("/")
def index():
    operation = request.args.get("operation", "add")
    if operation not in OPERATIONS:
        operation = "add"

    try:
        scalar = float(request.args.get("scalar", "2"))
        if not np.isfinite(scalar) or scalar == 0 and operation == "divide":
            raise ValueError
    except ValueError:
        scalar = 2.0

    try:
        sample_count = int(request.args.get("samples", "5"))
    except ValueError:
        sample_count = 5
    sample_count = min(max(sample_count, 3), 12)

    source = FEATURES[:sample_count]
    result = transform(source, operation, scalar)
    rows = [
        {
            "index": index + 1,
            "species": CLASS_NAMES[int(iris.target[index])],
            "source": [round(float(value), 2) for value in source[index]],
            "result": [round(float(value), 2) for value in result[index]],
        }
        for index in range(sample_count)
    ]
    means = np.mean(FEATURES, axis=0)

    return render_template(
        "index.html",
        operation=operation,
        operation_name=OPERATIONS[operation][0],
        operation_symbol=OPERATIONS[operation][1],
        scalar=scalar,
        sample_count=sample_count,
        operations=OPERATIONS,
        feature_names=FEATURE_NAMES,
        rows=rows,
        means=[round(float(value), 2) for value in means],
        sample_total=len(FEATURES),
        feature_total=FEATURES.shape[1],
        class_total=len(CLASS_NAMES),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
