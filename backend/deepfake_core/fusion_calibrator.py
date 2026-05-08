"""
训练后融合（fusion）校准器

输入 CSV（或预先导出的预测汇总），包含列：
  - video_id
  - video_fake_prob  (模型 A 给出的 P(fake))
  - frequency_score  (Module B, 可为空)
  - sync_score       (Module B, 可为空)
  - label            (字符串 'fake'/'real' 或数值类索引)

输出：保存训练好的校准器到指定路径（默认 models/fusion_calibrator.pkl）

使用方法示例：
  python -m src.fusion_calibrator --input outputs/metrics/calibration_dataset.csv --save models/fusion_calibrator.pkl

"""
from pathlib import Path
import argparse
import logging
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from . import config
from .utils import setup_logger


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # 支持 label 字段为字符串或数值
    if 'label' not in df.columns and 'is_fake' not in df.columns:
        raise ValueError('输入 CSV 必须包含 label 或 is_fake 列')

    if 'is_fake' in df.columns:
        y = df['is_fake'].astype(int)
    else:
        # 将 'fake' -> 1, 'real' -> 0
        y = df['label'].apply(lambda v: 1 if str(v).lower() == 'fake' else 0)

    # 特征列
    df_feat = pd.DataFrame()
    df_feat['video_fake_prob'] = df['video_fake_prob'].astype(float)
    # frequency_score / sync_score 允许空值
    df_feat['frequency_score'] = pd.to_numeric(df.get('frequency_score'), errors='coerce')
    df_feat['sync_score'] = pd.to_numeric(df.get('sync_score'), errors='coerce')
    # 缺失标志
    df_feat['frequency_missing'] = df_feat['frequency_score'].isna().astype(int)
    df_feat['sync_missing'] = df_feat['sync_score'].isna().astype(int)

    X = df_feat[['video_fake_prob', 'frequency_score', 'sync_score', 'frequency_missing', 'sync_missing']]
    return X, y


def train_and_save(X, y, save_path: Path, test_size=0.2, random_state=42, model_name: str = 'logistic'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_features = ['video_fake_prob', 'frequency_score', 'sync_score']
    binary_features = ['frequency_missing', 'sync_missing']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ], remainder='passthrough'  # binary features passthrough
    )

    # choose classifier
    if model_name == 'logistic':
        clf_est = LogisticRegression(class_weight='balanced', max_iter=1000)
    elif model_name == 'rf':
        clf_est = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=random_state)
    elif model_name == 'xgb':
        if not _HAS_XGB:
            raise RuntimeError('xgboost is not available in this environment')
        clf_est = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    else:
        raise ValueError(f'Unsupported model_name: {model_name}')

    clf = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', clf_est)
    ])

    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float('nan')

    logging.info(f"Train class distribution: {Counter(y_train)}")
    logging.info(f"Test class distribution: {Counter(y_test)}")
    logging.info(f"Calibration model accuracy: {acc:.4f}, AUC: {auc:.4f}")
    logging.info("Classification report:\n" + classification_report(y_test, y_pred, zero_division=0))

    # 保存模型
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    logging.info(f"Calibration model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train fusion calibrator')
    parser.add_argument('--input', type=str, required=True, help='Calibration CSV path')
    parser.add_argument('--save', type=str, default=str(config.MODELS_DIR / 'fusion_calibrator.pkl'), help='保存校准器路径')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--model', type=str, default='logistic', choices=['logistic','rf','xgb'], help='Model type to train')

    args = parser.parse_args()

    setup_logger()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    df = load_dataset(csv_path)
    X, y = prepare_features(df)

    save_path = Path(args.save)
    train_and_save(X, y, save_path, test_size=args.test_size, random_state=args.random_state, model_name=args.model)


if __name__ == '__main__':
    main()
