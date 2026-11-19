import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluate import get_frozen_split, load_dataset


def test_train_test_no_overlap():
    """1. Индексы обучающей и тестовой выборок не пересекаются."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = get_frozen_split(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(y_train.index).isdisjoint(set(y_test.index))


def test_unknown_category_handling():
    """2. Цепочка обработки принимает строку с неизвестной категорией."""
    model = joblib.load("models/best_model.joblib")
    unknown_row = pd.DataFrame([{
        "Species": "UnknownExoticFish",
        "Length1": 25.0,
        "Length2": 27.0,
        "Length3": 30.0,
        "Height": 10.0,
        "Width": 4.5,
    }])
    pred = model.predict(unknown_row)
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


def test_prediction_shape_and_finiteness():
    """3. Число прогнозов равно числу входных строк, все значения конечны."""
    df = load_dataset()
    _, X_test, _, _ = get_frozen_split(df)
    model = joblib.load("models/best_model.joblib")

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))


def test_metrics_match_manual_calculation():
    """4. Метрики на зафиксированном контрольном примере совпадают с ручным расчётом."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 320.0, 380.0])

    manual_mse = np.mean((y_true - y_pred) ** 2)
    manual_mae = np.mean(np.abs(y_true - y_pred))
    manual_r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    assert np.isclose(mean_squared_error(y_true, y_pred), manual_mse)
    assert np.isclose(mean_absolute_error(y_true, y_pred), manual_mae)
    assert np.isclose(r2_score(y_true, y_pred), manual_r2)


def test_saved_model_reproducibility(tmp_path):
    """5. Сохранённая цепочка обработки после загрузки воспроизводит прогноз в заданном допуске."""
    df = load_dataset()
    _, X_test, _, _ = get_frozen_split(df)
    model = joblib.load("models/best_model.joblib")

    preds_original = model.predict(X_test)
    copy_path = tmp_path / "reloaded_model.joblib"
    joblib.dump(model, copy_path)

    reloaded_model = joblib.load(copy_path)
    preds_reloaded = reloaded_model.predict(X_test)

    assert np.allclose(preds_original, preds_reloaded, atol=1e-6)