"""Resumable rolling-origin backtesting with SQLite checkpoints."""

__all__ = ["CheckpointError", "CheckpointMismatchError", "run_backtest"]


import copy
import hashlib
import json
import pickle
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp

from .compat import DataFrame, pl_DataFrame
from .validation import validate_format

_FORMAT_VERSION = 1
_META_KEY = "run"
_CUTOFF_COL = "cutoff"
_ROW_IDX_COL = "__utilsforecast_backtest_row"
_INTERVAL_COL_PATTERN = re.compile(r"-(?:lo|hi)-\d+")

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    CREATE TABLE IF NOT EXISTS predictions (
        round INTEGER NOT NULL,
        model TEXT NOT NULL,
        payload BLOB NOT NULL,
        PRIMARY KEY (round, model)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rounds (
        round INTEGER PRIMARY KEY,
        completed_at TEXT NOT NULL,
        n_rows INTEGER NOT NULL
    )
    """,
)


class CheckpointError(RuntimeError):
    """Raised when a backtest checkpoint cannot be used."""


class CheckpointMismatchError(CheckpointError):
    """Raised when a checkpoint was created with different data or settings."""


def _hash_array(hasher, arr: np.ndarray) -> None:
    if arr.dtype == object:
        hasher.update(b"<object>")
        for value in arr:
            hasher.update(type(value).__qualname__.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(repr(value).encode("utf-8", errors="backslashreplace"))
            hasher.update(b"\xff")
    else:
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(np.ascontiguousarray(arr).tobytes())


def _fingerprint_data(df: DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(df.shape).encode("utf-8"))
    for name in df.columns:
        hasher.update(b"\x00")
        hasher.update(str(name).encode("utf-8"))
        serie = df[name]
        hasher.update(str(serie.dtype).encode("utf-8"))
        _hash_array(hasher, serie.to_numpy())
    return hasher.hexdigest()


def _json_fingerprint(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_run_meta(
    df: DataFrame,
    model_names: List[str],
    decl_id: str,
    config: Optional[Dict[str, Any]],
    split_params: Dict[str, Any],
) -> Dict[str, str]:
    return {
        "format_version": str(_FORMAT_VERSION),
        "decl_id": decl_id,
        "data": _fingerprint_data(df),
        "splits": _json_fingerprint(split_params),
        "models": _json_fingerprint(model_names),
        "config": _json_fingerprint(config),
    }


_MISMATCH_LABELS = {
    "format_version": "checkpoint format version",
    "decl_id": "decl_id",
    "data": "input data",
    "splits": "split parameters",
    "models": "model names",
    "config": "config",
}


def _ensure_meta_matches(stored: Dict[str, str], current: Dict[str, str]) -> None:
    mismatched = [
        label
        for key, label in _MISMATCH_LABELS.items()
        if stored.get(key) != current.get(key)
    ]
    if mismatched:
        raise CheckpointMismatchError(
            "The existing checkpoint was created with different settings and "
            f"cannot be reused (changed: {', '.join(mismatched)}). "
            "Delete the checkpoint file or use a different `checkpoint_path` "
            "to start a new run."
        )


def _validate_models(
    models: Dict[str, Any], id_col: str, time_col: str, target_col: str
) -> None:
    reserved = {id_col, time_col, target_col, _CUTOFF_COL, _ROW_IDX_COL}
    for name, model in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Model names must be non-empty strings.")
        if name in reserved:
            raise ValueError(
                f"Model name '{name}' collides with a reserved column name. "
                "Please rename the model."
            )
        if _INTERVAL_COL_PATTERN.search(name):
            raise ValueError(
                f"Model name '{name}' matches the pattern used for prediction "
                "interval columns ('-lo-<level>'/'-hi-<level>') and would be "
                "ignored by `evaluate`. Please rename the model."
            )
        if not callable(getattr(model, "fit", None)) or not callable(
            getattr(model, "predict", None)
        ):
            raise ValueError(
                f"Model '{name}' must have callable `fit` and `predict` methods."
            )


def _validate_predictions(
    preds: DataFrame,
    model_name: str,
    valid: DataFrame,
    id_col: str,
    time_col: str,
) -> DataFrame:
    """Ensure predictions align exactly with the validation keys.

    Returns the predictions frame reduced to [id_col, time_col, model_name].
    """
    frame_type = pd.DataFrame if isinstance(valid, pd.DataFrame) else pl_DataFrame
    if not isinstance(preds, frame_type):
        expected = f"{frame_type.__module__}.{frame_type.__qualname__}"
        got = f"{type(preds).__module__}.{type(preds).__qualname__}"
        raise ValueError(
            f"Model '{model_name}' must return a dataframe of the same type "
            f"as `df` ({expected}) from `predict`, got {got}."
        )
    missing_cols = sorted({id_col, time_col} - set(preds.columns))
    if missing_cols:
        raise ValueError(
            f"Predictions from model '{model_name}' are missing required "
            f"columns: {missing_cols}."
        )
    value_cols = [c for c in preds.columns if c not in (id_col, time_col)]
    if len(value_cols) != 1:
        raise ValueError(
            f"Predictions from model '{model_name}' must contain exactly one "
            f"prediction column besides '{id_col}' and '{time_col}', "
            f"got {value_cols}."
        )
    if value_cols[0] != model_name:
        preds = ufp.rename(preds, {value_cols[0]: model_name})
    preds = preds[[id_col, time_col, model_name]]
    values = preds[model_name]
    if isinstance(preds, pd.DataFrame):
        is_numeric = pd.api.types.is_numeric_dtype(values.dtype)
    else:
        is_numeric = values.dtype.is_numeric()
    if not is_numeric:
        raise ValueError(
            f"Predictions from model '{model_name}' must be numeric, "
            f"got '{values.dtype}'."
        )
    keys = preds[[id_col, time_col]]
    if isinstance(keys, pd.DataFrame):
        n_unique = keys.drop_duplicates().shape[0]
    else:
        n_unique = keys.unique().shape[0]
    if n_unique != preds.shape[0]:
        raise ValueError(
            f"Model '{model_name}' produced duplicated "
            f"({id_col}, {time_col}) keys."
        )
    valid_keys = valid[[id_col, time_col]]
    missing = ufp.anti_join(valid_keys, keys, on=[id_col, time_col])
    if missing.shape[0]:
        raise ValueError(
            f"Model '{model_name}' is missing predictions for "
            f"{missing.shape[0]} validation rows."
        )
    extra = ufp.anti_join(keys, valid_keys, on=[id_col, time_col])
    if extra.shape[0]:
        raise ValueError(
            f"Model '{model_name}' produced predictions for {extra.shape[0]} "
            f"unknown ({id_col}, {time_col}) keys."
        )
    return preds


def _ordered_left_join(
    left: DataFrame, right: DataFrame, on: Union[str, List[str]]
) -> DataFrame:
    """Left join that preserves the row order of `left`."""
    left = ufp.assign_columns(
        ufp.copy_if_pandas(left), _ROW_IDX_COL, np.arange(left.shape[0])
    )
    merged = ufp.join(left, right, on=on, how="left")
    merged = ufp.sort(merged, by=_ROW_IDX_COL)
    return ufp.drop_columns(merged, _ROW_IDX_COL)


def _assemble_round_frame(
    base: DataFrame,
    cutoffs: DataFrame,
    model_frames: Dict[str, DataFrame],
    model_names: List[str],
    id_col: str,
    time_col: str,
    target_col: str,
) -> DataFrame:
    out = _ordered_left_join(base, cutoffs, on=id_col)
    for name in model_names:
        out = _ordered_left_join(out, model_frames[name], on=[id_col, time_col])
    return out[[id_col, time_col, _CUTOFF_COL, target_col, *model_names]]


def _fit_predict_round(
    models: Dict[str, Any],
    train: DataFrame,
    valid: DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
) -> Dict[str, DataFrame]:
    X_valid = ufp.drop_columns(valid, target_col)
    model_frames = {}
    for name, model in models.items():
        try:
            fitted = copy.deepcopy(model)
        except Exception as e:
            raise ValueError(
                f"Model '{name}' could not be deep-copied before fitting: {e}"
            ) from e
        fitted.fit(train)
        preds = fitted.predict(X_valid)
        model_frames[name] = _validate_predictions(
            preds, name, valid, id_col, time_col
        )
    return model_frames


def _connect(checkpoint_path: Union[str, Path]) -> sqlite3.Connection:
    path = Path(checkpoint_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    with conn:
        for stmt in _SCHEMA:
            conn.execute(stmt)
    return conn


def _read_meta(conn: sqlite3.Connection) -> Optional[Dict[str, str]]:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (_META_KEY,)
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError as e:
        raise CheckpointError(
            f"The checkpoint metadata is corrupted: {e}. "
            "Delete the checkpoint file and start a new run."
        ) from e


def _write_meta(conn: sqlite3.Connection, meta: Dict[str, str]) -> None:
    with conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            (_META_KEY, json.dumps(meta, sort_keys=True)),
        )


def _completed_rounds(conn: sqlite3.Connection) -> set:
    return {row[0] for row in conn.execute("SELECT round FROM rounds")}


def _save_round(
    conn: sqlite3.Connection,
    round_idx: int,
    model_frames: Dict[str, DataFrame],
    n_rows: int,
) -> None:
    payloads = [
        (
            round_idx,
            name,
            sqlite3.Binary(pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)),
        )
        for name, frame in model_frames.items()
    ]
    # all model predictions and the round completion marker are written in a
    # single transaction, so a failure can only roll back the current round
    with conn:
        conn.execute("DELETE FROM predictions WHERE round = ?", (round_idx,))
        conn.executemany(
            "INSERT INTO predictions (round, model, payload) VALUES (?, ?, ?)",
            payloads,
        )
        conn.execute(
            "INSERT INTO rounds (round, completed_at, n_rows) VALUES (?, ?, ?)",
            (round_idx, datetime.now(timezone.utc).isoformat(), n_rows),
        )


def _load_round(
    conn: sqlite3.Connection, round_idx: int, model_names: List[str]
) -> Dict[str, DataFrame]:
    rows = conn.execute(
        "SELECT model, payload FROM predictions WHERE round = ?", (round_idx,)
    ).fetchall()
    found = {name for name, _ in rows}
    expected = set(model_names)
    if found != expected:
        raise CheckpointError(
            f"The checkpoint is corrupted: round {round_idx} is marked as "
            f"completed but contains predictions for models {sorted(found)} "
            f"instead of {sorted(expected)}. "
            "Delete the checkpoint file and start a new run."
        )
    model_frames = {}
    for name, payload in rows:
        try:
            model_frames[name] = pickle.loads(payload)
        except Exception as e:
            raise CheckpointError(
                f"The checkpoint is corrupted: could not load the predictions "
                f"for model '{name}' in round {round_idx}: {e}. "
                "Delete the checkpoint file and start a new run."
            ) from e
    return model_frames


def run_backtest(
    df: DataFrame,
    models: Dict[str, Any],
    n_windows: int,
    h: int,
    freq: Union[int, str, pd.offsets.BaseOffset],
    checkpoint_path: Union[str, Path],
    *,
    decl_id: str,
    config: Optional[Dict[str, Any]] = None,
    step_size: Optional[int] = None,
    input_size: Optional[int] = None,
    allow_partial_horizons: bool = False,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> DataFrame:
    """Rolling-origin backtest with resumable SQLite checkpoints.

    Splits `df` with `utilsforecast.processing.backtest_splits` and, for every
    round, fits each model on the training window and predicts the validation
    window. Each round's predictions (for all models) together with the round
    completion marker are persisted to a SQLite checkpoint in a single
    transaction, so an interrupted run keeps every fully completed round and
    only the round in progress is rolled back. Re-running with the same
    settings skips completed rounds and merges the recovered predictions.

    Args:
        df (pandas or polars DataFrame): Panel data with columns
            `[id_col, time_col, target_col]` (extra columns are allowed and
            are passed to the models). The `(id_col, time_col)` pairs must be
            unique.
        models (dict): Mapping of model name to model instance. Each model
            must implement `fit(train_df)` and `predict(X)`:

            - `fit` receives only the round's training segment (same dataframe
              type as `df`, including `target_col`).
            - `predict` receives only the round's validation rows with
              `target_col` removed, and must return a dataframe of the same
              type as `df` with `id_col`, `time_col` and exactly one
              prediction column. The `(id_col, time_col)` keys must match the
              validation rows exactly: duplicated, missing or extra keys are
              rejected and nothing is saved for the round.

            Models are deep-copied before each fit, so every round starts from
            the original (unfitted) state and the passed instances are never
            modified.
        n_windows (int): Number of rolling windows.
        h (int): Forecast horizon (validation size per window).
        freq (int, str or pandas offset): Frequency of the time column, as
            accepted by `backtest_splits`.
        checkpoint_path (str or Path): Path of the SQLite checkpoint file.
            Completed rounds are stored here and reused on later runs.
        decl_id (str): Stable identifier of this backtest declaration. It is
            stored in the checkpoint and must be changed (e.g. when the model
            implementations change) to invalidate previous results. Function
            identity is not inspected; this id is the invalidation mechanism.
        config (dict, optional): JSON-serializable configuration associated
            with the run. Changing it invalidates the checkpoint.
        step_size (int, optional): Step between consecutive windows.
            Defaults to `h`.
        input_size (int, optional): Maximum training size per window.
        allow_partial_horizons (bool): Allow validation windows shorter than
            `h`. Defaults to False.
        id_col (str): Column that identifies each serie. Defaults to
            'unique_id'.
        time_col (str): Column that identifies each timestep. Defaults to 'ds'.
        target_col (str): Column that contains the target. Defaults to 'y'.

    Returns:
        pandas or polars DataFrame: Forecasts with columns
            `[id_col, time_col, 'cutoff', target_col, *models]`, in the same
            dataframe type as `df`. It contains the actual target values and
            the cutoff of each round, so it can be passed directly to
            `utilsforecast.evaluation.evaluate`.

    Raises:
        CheckpointMismatchError: If the checkpoint exists but was created with
            different data, split parameters, model names, `decl_id` or
            `config`. Delete the checkpoint file or use a different
            `checkpoint_path` to start a new run.
        CheckpointError: If the checkpoint is corrupted.
    """
    validate_format(df, id_col=id_col, time_col=time_col, target_col=target_col)
    if _CUTOFF_COL in (id_col, time_col, target_col):
        raise ValueError(
            f"'{id_col}', '{time_col}' and '{target_col}' must be different "
            f"from '{_CUTOFF_COL}', which is reserved for the output."
        )
    if n_windows < 1 or h < 1:
        raise ValueError("`n_windows` and `h` must be positive integers.")
    if not isinstance(decl_id, str) or not decl_id:
        raise ValueError(
            "`decl_id` must be a non-empty string that stably identifies this "
            "backtest declaration."
        )
    models = dict(models)
    if not models:
        raise ValueError("`models` must contain at least one model.")
    _validate_models(models, id_col=id_col, time_col=time_col, target_col=target_col)
    if config is not None:
        try:
            json.dumps(config)
        except TypeError as e:
            raise ValueError(f"`config` must be JSON-serializable: {e}") from e
    keys_df = df[[id_col, time_col]]
    if isinstance(keys_df, pd.DataFrame):
        has_duplicated_keys = keys_df.duplicated().any()
    else:
        has_duplicated_keys = keys_df.is_duplicated().any()
    if has_duplicated_keys:
        raise ValueError(f"`df` contains duplicated ({id_col}, {time_col}) keys.")

    model_names = list(models)
    split_params = {
        "n_windows": n_windows,
        "h": h,
        "step_size": step_size,
        "input_size": input_size,
        "allow_partial_horizons": allow_partial_horizons,
        "freq": repr(freq),
        "id_col": id_col,
        "time_col": time_col,
        "target_col": target_col,
    }
    meta = _build_run_meta(df, model_names, decl_id, config, split_params)

    conn = _connect(checkpoint_path)
    try:
        stored_meta = _read_meta(conn)
        if stored_meta is None:
            _write_meta(conn, meta)
        else:
            _ensure_meta_matches(stored_meta, meta)
        completed = _completed_rounds(conn)
        frames = []
        splits = ufp.backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
            step_size=step_size,
            input_size=input_size,
            allow_partial_horizons=allow_partial_horizons,
        )
        for round_idx, (cutoffs, train, valid) in enumerate(splits):
            # snapshot the validation keys and actuals before fitting, so a
            # model mutating its inputs cannot corrupt the output
            base = ufp.copy_if_pandas(
                valid[[id_col, time_col, target_col]], deep=True
            )
            if round_idx in completed:
                model_frames = _load_round(conn, round_idx, model_names)
                for name in model_names:
                    try:
                        model_frames[name] = _validate_predictions(
                            model_frames[name], name, valid, id_col, time_col
                        )
                    except ValueError as e:
                        raise CheckpointError(
                            "The checkpoint is corrupted: the recovered "
                            f"predictions for round {round_idx} do not align "
                            f"with the validation data: {e}. "
                            "Delete the checkpoint file and start a new run."
                        ) from e
            else:
                model_frames = _fit_predict_round(
                    models, train, valid, id_col, time_col, target_col
                )
                _save_round(conn, round_idx, model_frames, n_rows=valid.shape[0])
            frames.append(
                _assemble_round_frame(
                    base, cutoffs, model_frames, model_names, id_col, time_col,
                    target_col,
                )
            )
        return cast(DataFrame, ufp.vertical_concat(frames, match_categories=False))
    finally:
        conn.close()
