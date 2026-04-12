"""
MLP 模型
双隐藏层全连接网络，兼容 Qlib Model 接口（鸭子类型）

实现 fit(dataset) / predict(dataset, segment) 方法，
使得 MLP 可以被 Qlib 的 SignalRecord / SigAnaRecord 正常调用。

注意：没有继承 qlib.model.base.Model，因为 Qlib 的 LGBModel 等
内置模型也未强制继承该基类。Qlib 只要求 model 对象具有
fit(dataset) 和 predict(dataset, segment) 方法即可。
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from utils.helpers import get_model_config

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 未安装，MLP 模型不可用")


if TORCH_AVAILABLE:
    class MLPNet(nn.Module):
        """双隐藏层 MLP 网络"""

        def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


class MLPModel:
    """MLP 模型，兼容 Qlib Model 接口

    实现 fit(dataset) 和 predict(dataset) 方法，
    使得 MLP 可以被 Qlib workflow（SignalRecord 等）正常调用。

    Qlib Model 接口要求：
    - fit(dataset, **kwargs): 从 DatasetH 获取数据并训练
    - predict(dataset, segment="test"): 返回预测值 Series/DataFrame
    """

    def __init__(self, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch: pip install torch")

        cfg = kwargs if kwargs else get_model_config().get("mlp", {})
        self.hidden_dims = cfg.get("hidden_dims", [256, 128])
        self.dropout = cfg.get("dropout", 0.3)
        self.lr = cfg.get("learning_rate", 0.001)
        self.batch_size = cfg.get("batch_size", 4096)
        self.epochs = cfg.get("epochs", 100)
        self.patience = cfg.get("early_stopping_patience", 10)
        self.weight_decay = cfg.get("weight_decay", 0.0001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, dataset, **kwargs):
        """从 Qlib DatasetH 获取数据并训练

        Args:
            dataset: Qlib DatasetH 实例
        """
        # 从 DatasetH 提取 train/valid 数据
        df_train = dataset.prepare("train", col_set=["feature", "label"])
        df_valid = dataset.prepare("valid", col_set=["feature", "label"])

        # 分离 feature 和 label
        x_train = df_train["feature"].values.astype(np.float32)
        y_train = df_train["label"].iloc[:, 0].values.astype(np.float32)

        x_valid = df_valid["feature"].values.astype(np.float32)
        y_valid = df_valid["label"].iloc[:, 0].values.astype(np.float32)

        # 处理 NaN
        x_train = np.nan_to_num(x_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        x_valid = np.nan_to_num(x_valid, nan=0.0)
        y_valid = np.nan_to_num(y_valid, nan=0.0)

        input_dim = x_train.shape[1]
        logger.info(f"MLP 训练: 特征维度={input_dim}, "
                     f"训练样本={len(x_train)}, 验证样本={len(x_valid)}")

        self.model = MLPNet(input_dim, self.hidden_dims, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.FloatTensor(x_train), torch.FloatTensor(y_train)
        )
        train_loader = TorchDataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_cnt = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # 验证
            val_loss = self._evaluate_loss(x_valid, y_valid, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} | "
                            f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            if patience_cnt >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        logger.info("MLP 训练完成")

    def predict(self, dataset, segment="test"):
        """预测，返回格式兼容 Qlib SignalRecord

        Args:
            dataset: Qlib DatasetH 实例
            segment: 数据段 ("train"/"valid"/"test")

        Returns:
            pd.Series，index 与 dataset 的 segment 对齐
        """
        df = dataset.prepare(segment, col_set=["feature"])
        x = df["feature"].values.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)

        self.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(self.device)
            pred = self.model(x_t).cpu().numpy()

        # 返回 Series，index 为 (datetime, instrument) MultiIndex
        return pd.Series(pred, index=df.index)

    def _evaluate_loss(self, x, y, criterion):
        self.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            pred = self.model(x_t)
            loss = criterion(pred, y_t)
        return loss.item()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, input_dim: int):
        self.model = MLPNet(input_dim, self.hidden_dims, self.dropout).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def get_mlp_model_config(config: Optional[dict] = None) -> Dict[str, Any]:
    """返回 MLP 模型的 Qlib 配置

    注意：MLP 使用自定义类（非 Qlib 内置），
    module_path 指向本项目的 model.mlp_model

    Returns:
        模型配置字典
    """
    config = config or get_model_config()
    mlp_cfg = config.get("mlp", {})

    model_config = {
        "class": "MLPModel",
        "module_path": "model.mlp_model",
        "kwargs": {
            "hidden_dims": mlp_cfg.get("hidden_dims", [256, 128]),
            "dropout": mlp_cfg.get("dropout", 0.3),
            "learning_rate": mlp_cfg.get("learning_rate", 0.001),
            "batch_size": mlp_cfg.get("batch_size", 4096),
            "epochs": mlp_cfg.get("epochs", 100),
            "early_stopping_patience": mlp_cfg.get("early_stopping_patience", 10),
            "weight_decay": mlp_cfg.get("weight_decay", 0.0001),
        },
    }

    logger.info(f"MLP 模型配置: hidden={mlp_cfg.get('hidden_dims', [256, 128])}")
    return model_config
