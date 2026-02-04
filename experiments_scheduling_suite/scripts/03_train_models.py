from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import joblib
import numpy as np
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.models.baselines import MLPBaseline, NaivePersistence
from experiments_scheduling_suite.src.models.informer import build_model as build_informer
from experiments_scheduling_suite.src.models.lstm import build_model as build_lstm
from experiments_scheduling_suite.src.models.tcn import build_model as build_tcn
from experiments_scheduling_suite.src.models.transformer import build_model as build_transformer
from experiments_scheduling_suite.src.models.xgboost import XGBoostModel
from experiments_scheduling_suite.src.train.callbacks import save_history
from experiments_scheduling_suite.src.train.trainer import TrainConfig, train_model, predict_model
from experiments_scheduling_suite.src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练多模型。")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--models", type=str, default=None, help="逗号分隔的模型名")
    return parser.parse_args()


def _load_config_used(run_id: str) -> dict:
    path = PROJECT_ROOT / "reports" / run_id / "config_used.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _load_model_configs(model_names: List[str]) -> List[dict]:
    configs = []
    for name in model_names:
        path = PROJECT_ROOT / "configs" / "models" / f"{name}.yaml"
        if path.exists():
            configs.append(yaml.safe_load(path.read_text()) or {})
    return configs


def main() -> None:
    args = parse_args()
    run_id = args.run_id

    processed_dir = PROJECT_ROOT / "data" / "processed" / run_id
    reports_dir = PROJECT_ROOT / "reports" / run_id
    preds_dir = reports_dir / "preds"
    models_dir = reports_dir / "models"
    logs_dir = reports_dir / "logs"
    preds_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(processed_dir / "dataset.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    t_test = data["t_test"]

    meta = yaml.safe_load((processed_dir / "metadata.json").read_text())
    horizons = meta["horizons"]
    base_freq = meta.get("base_freq", "1S")
    feature_cols = meta.get("feature_cols", [])
    target_col = meta.get("target_col")
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0

    cfg_used = _load_config_used(run_id)
    train_cfg = cfg_used.get("training", {})
    set_seed(int(cfg_used.get("run", {}).get("seed", 42)))

    cfg = TrainConfig(
        epochs=int(train_cfg.get("epochs", 10)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        lr=float(train_cfg.get("lr", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        device=str(train_cfg.get("device", "cpu")),
    )

    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        # 默认加载所有配置文件
        model_names = [p.stem for p in (PROJECT_ROOT / "configs" / "models").glob("*.yaml")]

    model_cfgs = _load_model_configs(model_names)

    input_dim = X_train.shape[-1]
    for model_cfg in model_cfgs:
        model_name = model_cfg.get("model", {}).get("name")
        if not model_name:
            continue

        if model_name == "lstm":
            model = build_lstm(model_cfg.get("model", {}), input_dim, len(horizons))
            history = train_model(model, X_train, y_train, X_val, y_val, cfg)
            preds = predict_model(model, X_test, cfg)
            torch_path = models_dir / f"{model_name}.pt"
            import torch

            torch.save(model.state_dict(), torch_path)
            save_history(history, logs_dir / f"{model_name}.json")

        elif model_name == "transformer":
            model = build_transformer(model_cfg.get("model", {}), input_dim, len(horizons))
            history = train_model(model, X_train, y_train, X_val, y_val, cfg)
            preds = predict_model(model, X_test, cfg)
            import torch

            torch.save(model.state_dict(), models_dir / f"{model_name}.pt")
            save_history(history, logs_dir / f"{model_name}.json")

        elif model_name == "informer":
            model = build_informer(model_cfg.get("model", {}), input_dim, len(horizons))
            history = train_model(model, X_train, y_train, X_val, y_val, cfg)
            preds = predict_model(model, X_test, cfg)
            import torch

            torch.save(model.state_dict(), models_dir / f"{model_name}.pt")
            save_history(history, logs_dir / f"{model_name}.json")

        elif model_name == "tcn":
            model = build_tcn(model_cfg.get("model", {}), input_dim, len(horizons))
            history = train_model(model, X_train, y_train, X_val, y_val, cfg)
            preds = predict_model(model, X_test, cfg)
            import torch

            torch.save(model.state_dict(), models_dir / f"{model_name}.pt")
            save_history(history, logs_dir / f"{model_name}.json")

        elif model_name == "naive":
            model = NaivePersistence()
            model.fit(X_train, y_train, target_idx=target_idx)
            preds = model.predict(X_test)

        elif model_name == "mlp":
            model = MLPBaseline(tuple(model_cfg.get("model", {}).get("hidden_sizes", [128, 64])), int(model_cfg.get("model", {}).get("max_iter", 300)))
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            joblib.dump(model, models_dir / f"{model_name}.joblib")

        elif model_name == "xgboost":
            mcfg = model_cfg.get("model", {})
            model = XGBoostModel(
                n_estimators=int(mcfg.get("n_estimators", 200)),
                max_depth=int(mcfg.get("max_depth", 4)),
                learning_rate=float(mcfg.get("learning_rate", 0.05)),
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            joblib.dump(model, models_dir / f"{model_name}.joblib")

        else:
            print(f"Skip unknown model: {model_name}")
            continue

        # 保存预测（按 horizon 分文件）
        # 为不同 horizon 生成对应的“目标时间戳”
        t_base = pd.to_datetime(t_test)
        try:
            step = pd.to_timedelta(base_freq)
        except Exception:
            step = pd.to_timedelta(str(base_freq).lower())
        for i, h in enumerate(horizons):
            t_h = t_base + step * int(h)
            out = np.column_stack([t_h.astype(str), y_test[:, i], preds[:, i]])
            out_path = preds_dir / f"{model_name}_h{h}.csv"
            np.savetxt(out_path, out, delimiter=",", fmt="%s", header="timestamp,y_true,y_pred", comments="")

        print(f"Trained {model_name}")


if __name__ == "__main__":
    main()
