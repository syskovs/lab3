import json
from dataclasses import asdict

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

SEED = 42

students = pd.read_csv(r"C:\\projects\\project_practicum\\lab_3\\lab3\\Student_Performance.csv")
students["Extracurricular Activities"] = students["Extracurricular Activities"].map(
    {"Yes": 1, "No": 0}
)
Xr = students.drop(columns=["Performance Index"]).to_numpy(dtype=float)
yr = students["Performance Index"].to_numpy(dtype=float)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)

reg_input = RegressionInput(Xr_tr, yr_tr, Xr_te, yr_te)
reg = train_linear_regression(reg_input)
test_mse = float(mean_squared_error(yr_te, reg.predictions))

german = pd.read_csv(r"C:\\projects\\project_practicum\\lab_3\\lab3\\german.csv", sep=";")
Xc = german.drop(columns=["Creditability"]).to_numpy(dtype=float)
yc = german["Creditability"].to_numpy(dtype=int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.2, random_state=SEED, stratify=yc
)

clf_input = ClassificationInput(Xc_tr, yc_tr, Xc_te, yc_te)
clf = train_logistic_classifier(clf_input)
own = asdict(classification_metrics(yc_te, clf.probabilities, clf.predictions))

baselines = asdict(compare_classical_models(clf_input))

results = {
    "linear_regression": {"test_mse": test_mse},
    "classification": {"own_logistic": own, **baselines},
}
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

def at(history, epoch):
    return history[min(epoch, len(history)) - 1]

names = {
    "own_logistic": "Собственная логистическая регрессия",
    "logistic_regression": "sklearn LogisticRegression",
    "decision_tree": "DecisionTreeClassifier (max_depth=5)",
    "knn": "KNeighborsClassifier (k=5)",
}
cls = results["classification"]
rows = "\n".join(
    f"| {names[k]} | {m['roc_auc']:.4f} | {m['accuracy']:.4f} | "
    f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
    for k, m in cls.items()
)
ranking = sorted(cls, key=lambda k: cls[k]["roc_auc"], reverse=True)
place = ranking.index("own_logistic") + 1
diff_lines = "\n".join(
    f"- против «{names[k]}»: ROC-AUC {own['roc_auc'] - cls[k]['roc_auc']:+.4f}, "
    f"F1 {own['f1'] - cls[k]['f1']:+.4f}"
    for k in ("logistic_regression", "decision_tree", "knn")
)

report = f"""# Отчёт по лабораторной №3: линейные модели

## Данные и подготовка

- `Student_Performance.csv`: {len(students)} строк, цель – `Performance Index`.
  Признак `Extracurricular Activities` закодирован как Yes→1, No→0.
  Разбиение train/test = 80/20, `random_state={SEED}` (train {len(yr_tr)}, test {len(yr_te)}).
- `german.csv` (разделитель `;`): {len(german)} строк, цель – `Creditability`
  (доля класса 1: {yc.mean():.3f}). Все 20 признаков используются как числовые
  в исходной кодировке. Разбиение 80/20 со стратификацией, `random_state={SEED}`
  (train {len(yc_tr)}, test {len(yc_te)}).
- Масштабирование: стандартизация (z-score) внутри обеих собственных моделей,
  среднее и стандартное отклонение считаются только по train. После обучения веса
  пересчитываются в исходный масштаб признаков, поэтому прогноз равен `X @ weights + bias`.

## Гипотезы и эксперименты

1. Без стандартизации градиентный спуск с заданным шагом расходится (признаки в
   масштабах от единиц до тысяч), поэтому масштабирование встроено в обучение.
2. Линейная регрессия: пакетный градиентный спуск по MSE,
   `learning_rate={reg_input.learning_rate}`, `epochs={reg_input.epochs}`, нулевая инициализация.
   Ход MSE на train: эпоха 1 – {at(reg.loss_history, 1):.4f}, эпоха 10 – {at(reg.loss_history, 10):.4f},
   эпоха 100 – {at(reg.loss_history, 100):.4f}, последняя – {reg.loss_history[-1]:.4f}.
3. Логистическая регрессия: пакетный градиентный спуск по бинарной кросс-энтропии,
   `learning_rate={clf_input.learning_rate}`, `epochs={clf_input.epochs}`, порог 0.5.
   Ход BCE на train: эпоха 1 – {at(clf.loss_history, 1):.4f}, эпоха 10 – {at(clf.loss_history, 10):.4f},
   эпоха 100 – {at(clf.loss_history, 100):.4f}, последняя – {clf.loss_history[-1]:.4f}.
4. На том же разбиении обучены готовые LogisticRegression, DecisionTreeClassifier
   и KNeighborsClassifier (для LR и KNN – StandardScaler в пайплайне).

## Метрики

Линейная регрессия, MSE на тесте: **{test_mse:.4f}**.

Классификация (тестовая выборка `german.csv`):

| Модель | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
{rows}

## Выводы

- Линейная регрессия: MSE на train снизилась с {reg.loss_history[0]:.4f} до {reg.loss_history[-1]:.4f},
  MSE на тесте {test_mse:.4f}.
- Собственная логистическая регрессия занимает {place}-е место из 4 по ROC-AUC
  (лучшая модель: «{names[ranking[0]]}», ROC-AUC {cls[ranking[0]]['roc_auc']:.4f}).
  Разница собственной модели с остальными:
{diff_lines}
- Отличие от sklearn LogisticRegression объясняется тем, что там по умолчанию есть
  L2-регуляризация и другой оптимизатор (lbfgs), а собственная модель обучается
  без регуляризации фиксированное число эпох.
- Дерево и KNN – нелинейные модели. Дерево может ловить пороговые зависимости в
  категориальных кодах, но склонно к переобучению на 800 объектах; KNN страдает
  от 20-мерного пространства и от того, что категориальные коды трактуются как расстояния.
- Классы несбалансированы (≈70/30), поэтому Accuracy и Recall класса 1 завышены
  относительно качества разделения; основная метрика сравнения – ROC-AUC.

## Воспроизводимость

- Запуск: `python run_experiments.py` из корня проекта; скрипт пересоздаёт
  `experiment_results.json` и этот отчёт.
- Зафиксировано: `random_state={SEED}` для разбиений и дерева; собственные модели
  детерминированы (нулевая инициализация, пакетный спуск).
- Зависимости: numpy, pandas, scikit-learn.
"""
with open("REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)

print(json.dumps(results, ensure_ascii=False, indent=2))