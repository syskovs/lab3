import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _as_matrix(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _as_vector(y) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def _fit_scaler(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # защита от константных признаков
    return mean, std


def _to_original_scale(w_scaled: np.ndarray, b_scaled: float, mean, std):
    w = w_scaled / std
    b = float(b_scaled - np.sum(w_scaled * mean / std))
    return w, b


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    x_train = _as_matrix(data.train_features)
    y_train = _as_vector(data.train_target)
    x_test = _as_matrix(data.test_features)

    mean, std = _fit_scaler(x_train)
    xs = (x_train - mean) / std

    n, d = xs.shape
    w = np.zeros(d)
    b = 0.0
    loss_history: list[float] = []

    for _ in range(int(data.epochs)):
        error = xs @ w + b - y_train
        loss_history.append(float(np.mean(error ** 2)))
        w -= data.learning_rate * (2.0 / n) * (xs.T @ error)
        b -= data.learning_rate * (2.0 / n) * float(error.sum())

    weights, bias = _to_original_scale(w, b, mean, std)
    predictions = x_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    x_train = _as_matrix(data.train_features)
    y_train = _as_vector(data.train_target)
    x_test = _as_matrix(data.test_features)

    mean, std = _fit_scaler(x_train)
    xs = (x_train - mean) / std

    n, d = xs.shape
    w = np.zeros(d)
    b = 0.0
    eps = 1e-12
    loss_history: list[float] = []

    for _ in range(int(data.epochs)):
        p = _sigmoid(xs @ w + b)
        p_safe = np.clip(p, eps, 1.0 - eps)
        loss = -np.mean(y_train * np.log(p_safe) + (1.0 - y_train) * np.log(1.0 - p_safe))
        loss_history.append(float(loss))
        error = p - y_train
        w -= data.learning_rate * (xs.T @ error) / n
        b -= data.learning_rate * float(error.mean())

    weights, bias = _to_original_scale(w, b, mean, std)
    probabilities = _sigmoid(x_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def classification_metrics(y_true, probabilities, predictions) -> ClassificationMetrics:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    predictions = np.asarray(predictions).reshape(-1).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    x_train = _as_matrix(data.train_features)
    y_train = _as_vector(data.train_target).astype(int)
    x_test = _as_matrix(data.test_features)
    y_test = _as_vector(data.test_target).astype(int)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        pred = model.predict(x_test)
        results[name] = classification_metrics(y_test, proba, pred)

    return BaselineComparisonResult(**results)