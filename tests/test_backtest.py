import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from utilsforecast.backtest import (
    CheckpointError,
    CheckpointMismatchError,
    run_backtest,
)
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae


class CountingModel:
    """Base class that records fit/predict calls in class-level state.

    The executor deep-copies models before fitting, so only class-level
    state survives across rounds and runs.
    """

    n_fits = 0
    n_predicts = 0
    fit_frames = []
    predict_frames = []

    @classmethod
    def reset(cls):
        cls.n_fits = 0
        cls.n_predicts = 0
        cls.fit_frames = []
        cls.predict_frames = []


class LastValueModel(CountingModel):
    """Predicts the last observed value of each serie."""

    def fit(self, df):
        type(self).n_fits += 1
        type(self).fit_frames.append(df)
        if isinstance(df, pd.DataFrame):
            ordered = df.sort_values(["unique_id", "ds"])
            self.last_ = ordered.groupby("unique_id")["y"].last().to_dict()
        else:
            ordered = df.sort(["unique_id", "ds"])
            grouped = ordered.group_by("unique_id", maintain_order=True).agg(
                pl.col("y").last()
            )
            self.last_ = dict(
                zip(grouped["unique_id"].to_list(), grouped["y"].to_list())
            )

    def predict(self, X):
        type(self).n_predicts += 1
        type(self).predict_frames.append(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                {
                    "unique_id": X["unique_id"].to_numpy(),
                    "ds": X["ds"].to_numpy(),
                    "y_hat": X["unique_id"].map(self.last_).to_numpy("float64"),
                }
            )
        return pl.DataFrame(
            {
                "unique_id": X["unique_id"].to_list(),
                "ds": X["ds"].to_list(),
                "y_hat": [float(self.last_[uid]) for uid in X["unique_id"].to_list()],
            }
        )


class MeanModel(CountingModel):
    """Predicts the mean of the training segment of each serie."""

    def fit(self, df):
        type(self).n_fits += 1
        type(self).fit_frames.append(df)
        if isinstance(df, pd.DataFrame):
            self.mean_ = df.groupby("unique_id")["y"].mean().to_dict()
        else:
            grouped = df.group_by("unique_id").agg(pl.col("y").mean())
            self.mean_ = dict(
                zip(grouped["unique_id"].to_list(), grouped["y"].to_list())
            )

    def predict(self, X):
        type(self).n_predicts += 1
        type(self).predict_frames.append(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                {
                    "unique_id": X["unique_id"].to_numpy(),
                    "ds": X["ds"].to_numpy(),
                    "y_hat": X["unique_id"].map(self.mean_).to_numpy("float64"),
                }
            )
        return pl.DataFrame(
            {
                "unique_id": X["unique_id"].to_list(),
                "ds": X["ds"].to_list(),
                "y_hat": [float(self.mean_[uid]) for uid in X["unique_id"].to_list()],
            }
        )


class CrashOnceModel(CountingModel):
    """Raises on the first fit whose training data reaches `fail_train_max`."""

    crashed = False

    @classmethod
    def reset(cls):
        super().reset()
        cls.crashed = False

    def __init__(self, fail_train_max):
        self.fail_train_max = fail_train_max

    def fit(self, df):
        type(self).n_fits += 1
        if not type(self).crashed and df["ds"].max() >= self.fail_train_max:
            type(self).crashed = True
            raise RuntimeError("simulated crash")
        self.mean_ = df.groupby("unique_id")["y"].mean().to_dict()

    def predict(self, X):
        type(self).n_predicts += 1
        return pd.DataFrame(
            {
                "unique_id": X["unique_id"].to_numpy(),
                "ds": X["ds"].to_numpy(),
                "y_hat": X["unique_id"].map(self.mean_).to_numpy("float64"),
            }
        )


def make_pd(n_ids=2, n=20):
    frames = []
    for i in range(n_ids):
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"id_{i}",
                    "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                    "y": np.arange(n, dtype="float64") + 100.0 * i,
                    "x": np.arange(n, dtype="float64"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_pl(n_ids=2, n=20):
    start = datetime(2020, 1, 1)
    frames = []
    for i in range(n_ids):
        frames.append(
            pl.DataFrame(
                {
                    "unique_id": [f"id_{i}"] * n,
                    "ds": [start + timedelta(days=j) for j in range(n)],
                    "y": [float(j) + 100.0 * i for j in range(n)],
                    "x": [float(j) for j in range(n)],
                }
            )
        )
    return pl.concat(frames)


def read_completed_rounds(path):
    conn = sqlite3.connect(path)
    try:
        rounds = {row[0] for row in conn.execute("SELECT round FROM rounds")}
        pred_rounds = {
            row[0] for row in conn.execute("SELECT DISTINCT round FROM predictions")
        }
    finally:
        conn.close()
    return rounds, pred_rounds


def check_result_values(merged, id_idx):
    """Assert cutoffs, actuals and predictions of the naive/mean models."""
    first_round = merged["ds"] <= pd.Timestamp("2020-01-17")
    expected_cutoff = pd.Series(
        pd.Timestamp("2020-01-17"), index=merged.index
    )
    expected_cutoff[first_round] = pd.Timestamp("2020-01-14")
    pd.testing.assert_series_equal(
        merged["cutoff"], expected_cutoff, check_names=False, check_dtype=False
    )
    expected_naive = 100.0 * id_idx + np.where(first_round, 13.0, 16.0)
    np.testing.assert_allclose(merged["naive"], expected_naive)
    expected_mean = 100.0 * id_idx + np.where(first_round, 6.5, 8.0)
    np.testing.assert_allclose(merged["mean"], expected_mean)


def test_run_backtest_pandas(tmp_path):
    df = make_pd()
    LastValueModel.reset()
    MeanModel.reset()
    result = run_backtest(
        df,
        {"naive": LastValueModel(), "mean": MeanModel()},
        n_windows=2,
        h=3,
        freq="D",
        checkpoint_path=tmp_path / "ckpt.sqlite",
        decl_id="exp-1",
        config={"library": "test"},
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["unique_id", "ds", "cutoff", "y", "naive", "mean"]
    assert len(result) == 2 * 2 * 3  # n_ids * n_windows * h
    # actual target values and cutoffs are carried over
    merged = result.merge(
        df[["unique_id", "ds", "y"]].rename(columns={"y": "y_true"}),
        on=["unique_id", "ds"],
        how="left",
    )
    assert merged["y"].equals(merged["y_true"])
    id_idx = merged["unique_id"].str[-1].astype(int)
    check_result_values(merged, id_idx)
    # the output can be evaluated directly
    eval_result = evaluate(result, metrics=[mae], models=["naive", "mean"])
    mae_rows = eval_result[eval_result["metric"] == "mae"]
    assert (mae_rows["naive"] == 2.0).all()
    # evaluate groups by (id, cutoff): round-level MAEs for the mean model
    assert sorted(mae_rows["mean"].unique()) == [8.5, 10.0]


def test_run_backtest_polars(tmp_path):
    df = make_pl()
    LastValueModel.reset()
    MeanModel.reset()
    result = run_backtest(
        df,
        {"naive": LastValueModel(), "mean": MeanModel()},
        n_windows=2,
        h=3,
        freq="1d",
        checkpoint_path=tmp_path / "ckpt.sqlite",
        decl_id="exp-pl",
    )
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["unique_id", "ds", "cutoff", "y", "naive", "mean"]
    assert len(result) == 2 * 2 * 3
    merged = result.join(
        df.select(["unique_id", "ds", pl.col("y").alias("y_true")]),
        on=["unique_id", "ds"],
        how="left",
    )
    assert merged["y"].equals(merged["y_true"])
    pdf = merged.to_pandas()
    id_idx = pdf["unique_id"].str[-1].astype(int)
    check_result_values(pdf, id_idx)
    # the output can be evaluated directly in polars
    eval_result = evaluate(result, metrics=[mae], models=["naive", "mean"])
    assert isinstance(eval_result, pl.DataFrame)
    mae_rows = eval_result.filter(pl.col("metric") == "mae")
    assert (mae_rows["naive"] == 2.0).all()


def test_fit_and_predict_inputs_are_isolated(tmp_path):
    df = make_pd()
    LastValueModel.reset()
    model = LastValueModel()
    run_backtest(
        df,
        {"naive": model},
        n_windows=2,
        h=3,
        freq="D",
        checkpoint_path=tmp_path / "ckpt.sqlite",
        decl_id="iso",
    )
    assert LastValueModel.n_fits == 2
    assert LastValueModel.n_predicts == 2
    expected_cutoffs = [pd.Timestamp("2020-01-14"), pd.Timestamp("2020-01-17")]
    expected_train_sizes = [14 * 2, 17 * 2]
    for i in range(2):
        train = LastValueModel.fit_frames[i]
        X = LastValueModel.predict_frames[i]
        # fit only receives the training segment, target included
        assert "y" in train.columns
        assert train["ds"].max() == expected_cutoffs[i]
        assert len(train) == expected_train_sizes[i]
        # predict only receives the validation rows, target removed
        assert "y" not in X.columns
        assert "x" in X.columns  # exogenous columns are preserved
        assert X["ds"].min() > expected_cutoffs[i]
        assert len(X) == 3 * 2
    # the original instance is never fitted (models are deep-copied per round)
    assert not hasattr(model, "last_")


def _good_preds(X):
    return pd.DataFrame(
        {
            "unique_id": X["unique_id"].to_numpy(),
            "ds": X["ds"].to_numpy(),
            "y_hat": np.zeros(len(X)),
        }
    )


class DuplicateKeysModel:
    def fit(self, df):
        pass

    def predict(self, X):
        preds = _good_preds(X)
        return pd.concat([preds, preds.iloc[[0]]], ignore_index=True)


class MissingKeysModel:
    def fit(self, df):
        pass

    def predict(self, X):
        return _good_preds(X).iloc[1:]


class ExtraKeysModel:
    def fit(self, df):
        pass

    def predict(self, X):
        preds = _good_preds(X)
        extra = preds.iloc[[0]].copy()
        extra["ds"] = extra["ds"] + pd.Timedelta(days=100)
        return pd.concat([preds, extra], ignore_index=True)


class TwoValueColsModel:
    def fit(self, df):
        pass

    def predict(self, X):
        return _good_preds(X).assign(y_hat2=1.0)


class MissingTimeColModel:
    def fit(self, df):
        pass

    def predict(self, X):
        return _good_preds(X).drop(columns="ds")


class NonNumericModel:
    def fit(self, df):
        pass

    def predict(self, X):
        preds = _good_preds(X)
        preds["y_hat"] = "zero"
        return preds


class WrongTypeModel:
    def fit(self, df):
        pass

    def predict(self, X):
        return pl.DataFrame(
            {
                "unique_id": X["unique_id"].to_list(),
                "ds": X["ds"].to_list(),
                "y_hat": [0.0] * len(X),
            }
        )


@pytest.mark.parametrize(
    "bad_model,match",
    [
        (DuplicateKeysModel(), "duplicated"),
        (MissingKeysModel(), "missing predictions"),
        (ExtraKeysModel(), "unknown"),
        (TwoValueColsModel(), "exactly one prediction column"),
        (MissingTimeColModel(), "missing required columns"),
        (NonNumericModel(), "must be numeric"),
        (WrongTypeModel(), "same type"),
    ],
)
def test_invalid_predictions_are_rejected(tmp_path, bad_model, match):
    df = make_pd()
    ckpt = tmp_path / "ckpt.sqlite"
    with pytest.raises(ValueError, match=match):
        run_backtest(
            df,
            {"bad": bad_model},
            n_windows=2,
            h=3,
            freq="D",
            checkpoint_path=ckpt,
            decl_id="exp",
        )
    # nothing was committed for the failed round
    rounds, pred_rounds = read_completed_rounds(ckpt)
    assert rounds == set()
    assert pred_rounds == set()


def test_resume_after_crash_skips_completed_rounds(tmp_path):
    df = make_pd()
    ckpt = tmp_path / "ckpt.sqlite"
    LastValueModel.reset()
    CrashOnceModel.reset()
    models = {
        "naive": LastValueModel(),
        "crash": CrashOnceModel(fail_train_max=pd.Timestamp("2020-01-17")),
    }
    kwargs = dict(
        n_windows=2,
        h=3,
        freq="D",
        checkpoint_path=ckpt,
        decl_id="exp",
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_backtest(df, models, **kwargs)
    # round 0 was committed, round 1 was rolled back
    rounds, pred_rounds = read_completed_rounds(ckpt)
    assert rounds == {0}
    assert pred_rounds == {0}
    assert LastValueModel.n_fits == 2  # rounds 0 and 1
    assert CrashOnceModel.n_fits == 2

    # second run: the crash already happened, so the run completes
    result = run_backtest(df, models, **kwargs)
    assert LastValueModel.n_fits == 3  # only round 1 was (re)fitted
    assert CrashOnceModel.n_fits == 3
    assert len(result) == 2 * 2 * 3

    # the merged result equals an uninterrupted run
    reference = run_backtest(
        df, models, **{**kwargs, "checkpoint_path": tmp_path / "ref.sqlite"}
    )
    pd.testing.assert_frame_equal(result, reference)


def test_failed_save_rolls_back_only_current_round(tmp_path, monkeypatch):
    df = make_pd()
    ckpt = tmp_path / "ckpt.sqlite"
    LastValueModel.reset()
    real_connect = sqlite3.connect
    state = {"round_inserts": 0}

    class ConnProxy:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, parameters=()):
            if sql.lstrip().upper().startswith("INSERT INTO ROUNDS"):
                state["round_inserts"] += 1
                if state["round_inserts"] == 2:
                    raise RuntimeError("injected save failure")
            return self._real.execute(sql, parameters)

        def executemany(self, sql, parameters):
            return self._real.executemany(sql, parameters)

        def __enter__(self):
            self._real.__enter__()
            return self

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

        def close(self):
            return self._real.close()

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: ConnProxy(real_connect(*a, **k))
    )
    kwargs = dict(
        n_windows=2,
        h=3,
        freq="D",
        checkpoint_path=ckpt,
        decl_id="exp",
    )
    with pytest.raises(RuntimeError, match="injected save failure"):
        run_backtest(df, {"naive": LastValueModel()}, **kwargs)
    monkeypatch.undo()
    # round 0 survived; round 1's predictions were rolled back with its marker
    rounds, pred_rounds = read_completed_rounds(ckpt)
    assert rounds == {0}
    assert pred_rounds == {0}
    # the run can be resumed and completes
    result = run_backtest(df, {"naive": LastValueModel()}, **kwargs)
    assert len(result) == 2 * 2 * 3


def test_checkpoint_mismatch_is_rejected(tmp_path):
    df = make_pd()
    ckpt = tmp_path / "ckpt.sqlite"
    LastValueModel.reset()
    kwargs = dict(
        n_windows=2,
        h=3,
        freq="D",
        checkpoint_path=ckpt,
        decl_id="exp",
        config={"a": 1},
    )
    first = run_backtest(df, {"naive": LastValueModel()}, **kwargs)
    assert LastValueModel.n_fits == 2

    class ExplodingModel:
        def fit(self, df):
            raise AssertionError("fit should not be called")

        def predict(self, X):
            raise AssertionError("predict should not be called")

    # identical settings recover everything without refitting
    LastValueModel.reset()
    recovered = run_backtest(df, {"naive": ExplodingModel()}, **kwargs)
    pd.testing.assert_frame_equal(first, recovered)

    changed_df = df.assign(y=df["y"] + 1.0)
    with pytest.raises(CheckpointMismatchError, match="input data"):
        run_backtest(changed_df, {"naive": LastValueModel()}, **kwargs)
    with pytest.raises(CheckpointMismatchError, match="split parameters"):
        run_backtest(df, {"naive": LastValueModel()}, **{**kwargs, "h": 2})
    with pytest.raises(CheckpointMismatchError, match="split parameters"):
        run_backtest(df, {"naive": LastValueModel()}, **{**kwargs, "step_size": 1})
    with pytest.raises(CheckpointMismatchError, match="model names"):
        run_backtest(df, {"other": LastValueModel()}, **kwargs)
    with pytest.raises(CheckpointMismatchError, match="decl_id"):
        run_backtest(df, {"naive": LastValueModel()}, **{**kwargs, "decl_id": "new"})
    with pytest.raises(CheckpointMismatchError, match="config"):
        run_backtest(
            df, {"naive": LastValueModel()}, **{**kwargs, "config": {"a": 2}}
        )
    # a different checkpoint path starts a fresh run
    fresh = run_backtest(
        df,
        {"naive": LastValueModel()},
        **{**kwargs, "checkpoint_path": tmp_path / "other.sqlite", "decl_id": "new"},
    )
    pd.testing.assert_frame_equal(first, fresh)


def test_corrupted_checkpoint_is_rejected(tmp_path):
    df = make_pd()
    ckpt = tmp_path / "ckpt.sqlite"
    LastValueModel.reset()
    kwargs = dict(
        n_windows=2,
        h=3,
        freq="D",
        checkpoint_path=ckpt,
        decl_id="exp",
    )
    run_backtest(df, {"naive": LastValueModel()}, **kwargs)
    conn = sqlite3.connect(ckpt)
    conn.execute("DELETE FROM predictions WHERE round = 1")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError, match="corrupted"):
        run_backtest(df, {"naive": LastValueModel()}, **kwargs)


def test_shuffled_input_produces_aligned_results(tmp_path):
    df = make_pd()
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    models = {"naive": LastValueModel(), "mean": MeanModel()}
    kwargs = dict(n_windows=2, h=3, freq="D", decl_id="exp")
    sorted_result = run_backtest(
        df, models, checkpoint_path=tmp_path / "sorted.sqlite", **kwargs
    )
    shuffled_result = run_backtest(
        shuffled, models, checkpoint_path=tmp_path / "shuffled.sqlite", **kwargs
    )
    by = ["unique_id", "cutoff", "ds"]
    pd.testing.assert_frame_equal(
        sorted_result.sort_values(by).reset_index(drop=True),
        shuffled_result.sort_values(by).reset_index(drop=True),
    )


def test_overlapping_windows(tmp_path):
    df = make_pd()
    LastValueModel.reset()
    result = run_backtest(
        df,
        {"naive": LastValueModel()},
        n_windows=2,
        h=3,
        step_size=2,
        freq="D",
        checkpoint_path=tmp_path / "ckpt.sqlite",
        decl_id="exp",
    )
    assert len(result) == 2 * 2 * 3
    # validation windows overlap, so (id, ds) repeats across rounds...
    assert result.duplicated(["unique_id", "ds"]).any()
    # ...but (id, cutoff, ds) stays unique
    assert not result.duplicated(["unique_id", "cutoff", "ds"]).any()
    eval_result = evaluate(result, metrics=[mae], models=["naive"])
    # evaluate groups by (id, cutoff): 2 ids x 2 rounds
    assert len(eval_result[eval_result["metric"] == "mae"]) == 4


def test_integer_time_column(tmp_path):
    n = 20
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], n),
            "ds": np.tile(np.arange(n), 2),
            "y": np.tile(np.arange(n, dtype="float64"), 2),
        }
    )
    LastValueModel.reset()
    result = run_backtest(
        df,
        {"naive": LastValueModel()},
        n_windows=2,
        h=3,
        freq=1,
        checkpoint_path=tmp_path / "ckpt.sqlite",
        decl_id="exp",
    )
    assert pd.api.types.is_integer_dtype(result["ds"])
    assert set(result["cutoff"]) == {13, 16}
    assert len(result) == 2 * 2 * 3


def test_input_validation(tmp_path):
    df = make_pd()
    ckpt = tmp_path / "ckpt.sqlite"
    base = dict(n_windows=2, h=3, freq="D", checkpoint_path=ckpt)
    with pytest.raises(ValueError, match="decl_id"):
        run_backtest(df, {"m": LastValueModel()}, decl_id="", **base)
    with pytest.raises(ValueError, match="at least one model"):
        run_backtest(df, {}, decl_id="x", **base)
    with pytest.raises(ValueError, match="JSON-serializable"):
        run_backtest(
            df, {"m": LastValueModel()}, decl_id="x", config={"f": object()}, **base
        )
    with pytest.raises(ValueError, match="collides"):
        run_backtest(df, {"y": LastValueModel()}, decl_id="x", **base)
    with pytest.raises(ValueError, match="callable `fit` and `predict`"):
        run_backtest(df, {"m": object()}, decl_id="x", **base)
    duplicated = pd.concat([df, df.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        run_backtest(duplicated, {"m": LastValueModel()}, decl_id="x", **base)
