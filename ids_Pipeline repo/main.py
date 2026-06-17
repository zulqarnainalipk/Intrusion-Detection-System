# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Advanced Deep Learning Based Intrusion Detection System
# ## Blockchain Integrated Intrusion Detection Framework
# ---
# #
# ### Project Objectives
# #
# This project implements:
# #
# - NSL-KDD based Intrusion Detection System
# - Deep Learning based attack classification using Stacked LSTM + Attention
# - Blockchain integrated secure alert logging with threat levels
# - EDA with statistical analysis and advanced visualizations
# - Feature importance, PCA, t-SNE dimensionality reduction
# - Precision-Recall, ROC, and normalized confusion matrix analysis
# - Overfitting / Underfitting prevention via regularization + callbacks
# - Full GPU support with mixed precision training
# - Saved artifacts for deployment and web demo integration
# - Auto-generated HTML + CSV project report
# #
# ---
# #
# ### Technologies Used
# #
# - Python
# - TensorFlow / Keras (GPU + Mixed Precision)
# - Scikit-learn
# - Pandas / NumPy
# - Matplotlib / Seaborn / Plotly
# - Blockchain using Python (SHA-256)
# - SHAP / Feature Importance via Random Forest
# #
# ---
# #
# ### Dataset
# #
# Dataset: NSL-KDD
# #
# Dataset Path: `/kaggle/input/datasets/hassan06/nslkdd`
# #
# ---
# #
# ### Outputs Generated
# #
# - Trained LSTM model (.h5 + SavedModel)
# - Feature scaler + label encoders
# - Blockchain logs (CSV + JSON)
# - Full evaluation metrics (CSV)
# - Confusion matrix (raw + normalized)
# - Accuracy / Loss training curves
# - ROC curve + Precision-Recall curve
# - Correlation heatmap
# - Feature importance chart
# - Feature distributions by class
# - PCA 2D scatter
# - t-SNE 2D scatter
# - Attack category distribution
# - Binary class distribution
# - Per-epoch metrics log
# - HTML project report
# - Full summary JSON

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:37:57.489708Z","iopub.execute_input":"2026-05-10T16:37:57.490168Z","iopub.status.idle":"2026-05-10T16:38:27.212456Z","shell.execute_reply.started":"2026-05-10T16:37:57.490137Z","shell.execute_reply":"2026-05-10T16:38:27.211597Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 : IMPORTS & ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import hashlib
import warnings
import datetime
import time as time_module

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from time import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    BatchNormalization,
    Input,
    Bidirectional,
    GlobalAveragePooling1D,
    Multiply,
    Permute,
    Reshape,
    Flatten,
    Lambda,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

print("=" * 70)
print("  INITIALIZING PROJECT — ADVANCED IDS FRAMEWORK")
print("=" * 70)

print(f"TensorFlow Version : {tf.__version__}")
print(f"NumPy Version      : {np.__version__}")
print(f"Pandas Version     : {pd.__version__}")
print(f"Run Timestamp      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

sns.set_style("whitegrid")
sns.set_palette("husl")
PALETTE = sns.color_palette("husl", 10)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # GPU Configuration & Mixed Precision
# GPU memory growth is enabled to prevent OOM errors.
# Mixed precision (float16) is enabled when a GPU is available for 2-3x speedup.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:27.214103Z","iopub.execute_input":"2026-05-10T16:38:27.214689Z","iopub.status.idle":"2026-05-10T16:38:28.796455Z","shell.execute_reply.started":"2026-05-10T16:38:27.214664Z","shell.execute_reply":"2026-05-10T16:38:28.795395Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 : GPU CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  GPU CONFIGURATION")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Enable mixed precision for faster training on modern GPUs
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")

        print(f"GPUs Detected      : {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}            : {gpu.name}")
        print("Memory Growth      : Enabled")
        print("Mixed Precision    : float16 (enabled)")

    except RuntimeError as e:
        print(f"GPU Config Error   : {e}")

else:
    print("GPUs Detected      : None — running on CPU")
    print("Tip: Enable GPU accelerator in Kaggle Settings → Accelerator → GPU")

print(f"Logical Devices    : {tf.config.list_logical_devices()}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:28.797577Z","iopub.execute_input":"2026-05-10T16:38:28.798936Z","iopub.status.idle":"2026-05-10T16:38:28.824878Z","shell.execute_reply.started":"2026-05-10T16:38:28.798892Z","shell.execute_reply":"2026-05-10T16:38:28.824186Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 : OUTPUT DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR    = "project_outputs"
FIGURE_DIR    = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR    = os.path.join(OUTPUT_DIR, "reports")
BLOCKCHAIN_DIR= os.path.join(OUTPUT_DIR, "blockchain")
LOG_DIR       = os.path.join(OUTPUT_DIR, "logs")
EDA_DIR       = os.path.join(FIGURE_DIR, "eda")
EVAL_DIR      = os.path.join(FIGURE_DIR, "evaluation")
TRAIN_DIR     = os.path.join(FIGURE_DIR, "training")

for directory in [
    FIGURE_DIR, MODEL_DIR, REPORT_DIR,
    BLOCKCHAIN_DIR, LOG_DIR, EDA_DIR, EVAL_DIR, TRAIN_DIR
]:
    os.makedirs(directory, exist_ok=True)

print("Output directories created successfully")
print(f"Run ID             : {RUN_ID}")

# Master log dictionary — everything gets recorded here
MASTER_LOG = {
    "run_id"       : RUN_ID,
    "timestamp"    : datetime.datetime.now().isoformat(),
    "gpu_available": len(gpus) > 0,
    "gpu_count"    : len(gpus),
    "dataset"      : {},
    "preprocessing": {},
    "model"        : {},
    "training"     : {},
    "evaluation"   : {},
    "blockchain"   : {},
    "artifacts"    : [],
}

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Dataset Loading
# #
# The NSL-KDD dataset is loaded with full column naming.
# Both train and test splits are verified before proceeding.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:28.825940Z","iopub.execute_input":"2026-05-10T16:38:28.826224Z","iopub.status.idle":"2026-05-10T16:38:28.845197Z","shell.execute_reply.started":"2026-05-10T16:38:28.826192Z","shell.execute_reply":"2026-05-10T16:38:28.844653Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 : DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  LOADING DATASET")
print("=" * 70)

DATA_PATH  = "/kaggle/input/datasets/hassan06/nslkdd"
TRAIN_FILE = os.path.join(DATA_PATH, "KDDTrain+.txt")
TEST_FILE  = os.path.join(DATA_PATH, "KDDTest+.txt")

print(f"Train File Exists  : {os.path.exists(TRAIN_FILE)}")
print(f"Test File Exists   : {os.path.exists(TEST_FILE)}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:28.847288Z","iopub.execute_input":"2026-05-10T16:38:28.847997Z","iopub.status.idle":"2026-05-10T16:38:28.853066Z","shell.execute_reply.started":"2026-05-10T16:38:28.847975Z","shell.execute_reply":"2026-05-10T16:38:28.852473Z"}}
columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty"
]

print(f"Total Columns      : {len(columns)}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:28.853809Z","iopub.execute_input":"2026-05-10T16:38:28.854048Z","iopub.status.idle":"2026-05-10T16:38:29.462285Z","shell.execute_reply.started":"2026-05-10T16:38:28.854014Z","shell.execute_reply":"2026-05-10T16:38:29.461628Z"}}
train_df = pd.read_csv(TRAIN_FILE, names=columns)
test_df  = pd.read_csv(TEST_FILE,  names=columns)

print(f"Train Shape        : {train_df.shape}")
print(f"Test Shape         : {test_df.shape}")

MASTER_LOG["dataset"]["train_samples"] = len(train_df)
MASTER_LOG["dataset"]["test_samples"]  = len(test_df)
MASTER_LOG["dataset"]["total_features"] = len(columns) - 2  # minus attack + difficulty

display(train_df.head())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Exploratory Data Analysis (EDA)
# #
# Comprehensive dataset exploration including:
# #
# - Missing value analysis
# - Statistical summary (mean, std, min/max, skew, kurtosis)
# - Attack category distribution
# - Binary class distribution
# - Protocol and service analysis
# - Numerical feature distributions by class
# - Correlation heatmap
# - Outlier detection using IQR
# #
# All EDA figures are saved in high resolution for report use.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:29.463335Z","iopub.execute_input":"2026-05-10T16:38:29.463700Z","iopub.status.idle":"2026-05-10T16:38:29.502558Z","shell.execute_reply.started":"2026-05-10T16:38:29.463675Z","shell.execute_reply":"2026-05-10T16:38:29.502014Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 : MISSING VALUES & BASIC STATS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  MISSING VALUE ANALYSIS")
print("=" * 70)

train_missing = train_df.isnull().sum().sum()
test_missing  = test_df.isnull().sum().sum()

print(f"Train Missing Values : {train_missing}")
print(f"Test Missing Values  : {test_missing}")

MASTER_LOG["dataset"]["train_missing_values"] = int(train_missing)
MASTER_LOG["dataset"]["test_missing_values"]  = int(test_missing)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:29.503602Z","iopub.execute_input":"2026-05-10T16:38:29.503958Z","iopub.status.idle":"2026-05-10T16:38:29.743456Z","shell.execute_reply.started":"2026-05-10T16:38:29.503920Z","shell.execute_reply":"2026-05-10T16:38:29.742772Z"}}
print("=" * 70)
print("  STATISTICAL SUMMARY — NUMERICAL FEATURES")
print("=" * 70)

numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [c for c in numerical_cols if c not in ["land", "logged_in",
    "root_shell", "su_attempted", "is_host_login", "is_guest_login",
    "num_outbound_cmds", "difficulty"]]

stats_df = train_df[numerical_cols].describe().T
stats_df["skewness"]  = train_df[numerical_cols].skew()
stats_df["kurtosis"]  = train_df[numerical_cols].kurt()

stats_csv_path = os.path.join(REPORT_DIR, "feature_statistics.csv")
stats_df.to_csv(stats_csv_path)

display(stats_df.head(15))
print(f"Statistics Saved   : {stats_csv_path}")
MASTER_LOG["artifacts"].append(stats_csv_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:29.744547Z","iopub.execute_input":"2026-05-10T16:38:29.744820Z","iopub.status.idle":"2026-05-10T16:38:29.772002Z","shell.execute_reply.started":"2026-05-10T16:38:29.744782Z","shell.execute_reply":"2026-05-10T16:38:29.771347Z"}}
print("=" * 70)
print("  ATTACK DISTRIBUTION ANALYSIS")
print("=" * 70)

attack_distribution = train_df["attack"].value_counts()
attack_pct          = (attack_distribution / len(train_df) * 100).round(2)

attack_summary_df = pd.DataFrame({
    "Count"     : attack_distribution,
    "Percentage": attack_pct
})

attack_summary_path = os.path.join(REPORT_DIR, "attack_distribution.csv")
attack_summary_df.to_csv(attack_summary_path)

print(attack_summary_df.head(20))
print(f"Attack Summary Saved : {attack_summary_path}")
MASTER_LOG["dataset"]["unique_attack_types"] = int(train_df["attack"].nunique())
MASTER_LOG["artifacts"].append(attack_summary_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:29.773095Z","iopub.execute_input":"2026-05-10T16:38:29.773402Z","iopub.status.idle":"2026-05-10T16:38:31.501334Z","shell.execute_reply.started":"2026-05-10T16:38:29.773378Z","shell.execute_reply":"2026-05-10T16:38:31.500562Z"}}
# ── Figure 1 : Attack Category Distribution ───────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Bar chart — top 15
top_attacks = attack_distribution.head(15)
colors = sns.color_palette("husl", len(top_attacks))
bars = axes[0].bar(range(len(top_attacks)), top_attacks.values, color=colors, edgecolor="white", linewidth=0.5)
axes[0].set_xticks(range(len(top_attacks)))
axes[0].set_xticklabels(top_attacks.index, rotation=45, ha="right", fontsize=9)
axes[0].set_title("Top 15 Attack Categories", fontsize=14, fontweight="bold", pad=12)
axes[0].set_xlabel("Attack Type", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)

for bar, val in zip(bars, top_attacks.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:,}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# Horizontal bar — all attacks
all_attacks = attack_distribution
colors_all  = sns.color_palette("coolwarm", len(all_attacks))
axes[1].barh(range(len(all_attacks)), all_attacks.values, color=colors_all, edgecolor="white")
axes[1].set_yticks(range(len(all_attacks)))
axes[1].set_yticklabels(all_attacks.index, fontsize=8)
axes[1].set_title("All Attack Categories (Horizontal)", fontsize=14, fontweight="bold", pad=12)
axes[1].set_xlabel("Count", fontsize=11)
axes[1].invert_yaxis()

plt.suptitle("NSL-KDD Attack Distribution Analysis", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()

path = os.path.join(EDA_DIR, "attack_distribution.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:31.502232Z","iopub.execute_input":"2026-05-10T16:38:31.502437Z","iopub.status.idle":"2026-05-10T16:38:33.117178Z","shell.execute_reply.started":"2026-05-10T16:38:31.502418Z","shell.execute_reply":"2026-05-10T16:38:33.116325Z"}}
# ── Figure 2 : Binary Class & Protocol / Service / Flag Distribution ──────

train_df["label"] = train_df["attack"].apply(lambda x: 0 if x == "normal" else 1)
test_df["label"]  = test_df["attack"].apply(lambda x: 0 if x == "normal" else 1)

label_counts = train_df["label"].value_counts()

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# Pie — binary class
wedge_colors = ["#2ecc71", "#e74c3c"]
wedges, texts, autotexts = axes[0].pie(
    label_counts.values,
    labels=["Normal", "Attack"],
    autopct="%1.1f%%",
    colors=wedge_colors,
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight("bold")
axes[0].set_title("Binary Class\nDistribution", fontsize=12, fontweight="bold")

# Bar — protocol_type
proto_counts = train_df["protocol_type"].value_counts()
axes[1].bar(proto_counts.index, proto_counts.values,
            color=sns.color_palette("Set2", len(proto_counts)), edgecolor="white")
axes[1].set_title("Protocol Type\nDistribution", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Protocol")
axes[1].set_ylabel("Count")
for i, v in enumerate(proto_counts.values):
    axes[1].text(i, v + 100, f"{v:,}", ha="center", fontsize=9, fontweight="bold")

# Bar — top 10 services
top_services = train_df["service"].value_counts().head(10)
axes[2].barh(top_services.index, top_services.values,
             color=sns.color_palette("Set3", len(top_services)))
axes[2].set_title("Top 10 Services", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Count")
axes[2].invert_yaxis()

# Bar — flag
flag_counts = train_df["flag"].value_counts()
axes[3].bar(flag_counts.index, flag_counts.values,
            color=sns.color_palette("Spectral", len(flag_counts)), edgecolor="white")
axes[3].set_title("Flag Distribution", fontsize=12, fontweight="bold")
axes[3].set_xlabel("Flag")
axes[3].set_ylabel("Count")
axes[3].tick_params(axis="x", rotation=45)

plt.suptitle("Categorical Feature Analysis — NSL-KDD Dataset", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

path = os.path.join(EDA_DIR, "categorical_distributions.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:33.118245Z","iopub.execute_input":"2026-05-10T16:38:33.119014Z","iopub.status.idle":"2026-05-10T16:38:39.574929Z","shell.execute_reply.started":"2026-05-10T16:38:33.118985Z","shell.execute_reply":"2026-05-10T16:38:39.574079Z"}}
# ── Figure 3 : Numerical Feature Distributions by Class ──────────────────

print("=" * 70)
print("  FEATURE DISTRIBUTION BY CLASS")
print("=" * 70)

key_features = [
    "duration", "src_bytes", "dst_bytes", "count",
    "srv_count", "serror_rate", "rerror_rate", "same_srv_rate",
    "diff_srv_rate", "dst_host_count", "hot", "num_compromised"
]

fig, axes = plt.subplots(3, 4, figsize=(22, 14))
axes = axes.flatten()

normal_df = train_df[train_df["label"] == 0]
attack_df = train_df[train_df["label"] == 1]

for i, feature in enumerate(key_features):
    ax = axes[i]

    n_vals = normal_df[feature].clip(upper=normal_df[feature].quantile(0.99))
    a_vals = attack_df[feature].clip(upper=attack_df[feature].quantile(0.99))

    ax.hist(n_vals, bins=40, alpha=0.65, color="#2ecc71", label="Normal",   density=True, edgecolor="white")
    ax.hist(a_vals, bins=40, alpha=0.65, color="#e74c3c", label="Attack",   density=True, edgecolor="white")

    ax.set_title(feature.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

plt.suptitle("Feature Distributions: Normal vs Attack Traffic", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()

path = os.path.join(EDA_DIR, "feature_distributions_by_class.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:39.575902Z","iopub.execute_input":"2026-05-10T16:38:39.576214Z","iopub.status.idle":"2026-05-10T16:38:42.747228Z","shell.execute_reply.started":"2026-05-10T16:38:39.576192Z","shell.execute_reply":"2026-05-10T16:38:42.746144Z"}}
# ── Figure 4 : Correlation Heatmap ────────────────────────────────────────

print("=" * 70)
print("  CORRELATION HEATMAP")
print("=" * 70)

numerical_train = train_df.select_dtypes(include=[np.number]).drop(
    columns=["label", "difficulty"], errors="ignore"
)

corr_matrix = numerical_train.corr()

# Save correlation matrix to CSV
corr_path = os.path.join(REPORT_DIR, "correlation_matrix.csv")
corr_matrix.to_csv(corr_path)
print(f"Correlation CSV Saved : {corr_path}")
MASTER_LOG["artifacts"].append(corr_path)

fig, ax = plt.subplots(figsize=(22, 18))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=False,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    center=0,
    linewidths=0.3,
    ax=ax,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
)

ax.set_title("Feature Correlation Heatmap (NSL-KDD Numerical Features)",
             fontsize=16, fontweight="bold", pad=15)
ax.tick_params(axis="x", rotation=90, labelsize=7)
ax.tick_params(axis="y", rotation=0,  labelsize=7)

path = os.path.join(EDA_DIR, "correlation_heatmap.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:42.751217Z","iopub.execute_input":"2026-05-10T16:38:42.751546Z","iopub.status.idle":"2026-05-10T16:38:42.787033Z","shell.execute_reply.started":"2026-05-10T16:38:42.751522Z","shell.execute_reply":"2026-05-10T16:38:42.786370Z"}}
# ── Figure 5 : Highly Correlated Feature Pairs ────────────────────────────

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if abs(val) >= 0.80:
            high_corr_pairs.append({
                "Feature A": corr_matrix.columns[i],
                "Feature B": corr_matrix.columns[j],
                "Correlation": round(val, 4)
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("Correlation", key=abs, ascending=False)
high_corr_path = os.path.join(REPORT_DIR, "high_correlation_pairs.csv")
high_corr_df.to_csv(high_corr_path, index=False)
print(f"High Correlation Pairs (|r| ≥ 0.80) : {len(high_corr_df)}")
display(high_corr_df.head(15))
MASTER_LOG["dataset"]["high_correlation_pairs"] = len(high_corr_df)
MASTER_LOG["artifacts"].append(high_corr_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:42.788176Z","iopub.execute_input":"2026-05-10T16:38:42.788469Z","iopub.status.idle":"2026-05-10T16:38:49.323936Z","shell.execute_reply.started":"2026-05-10T16:38:42.788447Z","shell.execute_reply":"2026-05-10T16:38:49.323115Z"}}
# ── Figure 6 : Outlier Detection (IQR Method) ─────────────────────────────

print("=" * 70)
print("  OUTLIER ANALYSIS")
print("=" * 70)

outlier_report = []

for col in key_features:
    Q1  = train_df[col].quantile(0.25)
    Q3  = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((train_df[col] < lower) | (train_df[col] > upper)).sum()
    outlier_report.append({
        "Feature"     : col,
        "Q1"          : round(Q1, 4),
        "Q3"          : round(Q3, 4),
        "IQR"         : round(IQR, 4),
        "Lower Bound" : round(lower, 4),
        "Upper Bound" : round(upper, 4),
        "Outliers"    : int(n_outliers),
        "Outlier %"   : round(n_outliers / len(train_df) * 100, 2)
    })

outlier_df = pd.DataFrame(outlier_report)
outlier_path = os.path.join(REPORT_DIR, "outlier_analysis.csv")
outlier_df.to_csv(outlier_path, index=False)

display(outlier_df)
print(f"Outlier Report Saved : {outlier_path}")
MASTER_LOG["artifacts"].append(outlier_path)

# Box plot for outliers
fig, axes = plt.subplots(2, 6, figsize=(24, 10))
axes = axes.flatten()

for i, col in enumerate(key_features):
    sns.boxplot(
        data=train_df,
        x="label",
        y=col,
        palette=["#2ecc71", "#e74c3c"],
        ax=axes[i],
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3}
    )
    axes[i].set_title(col.replace("_", " ").title(), fontsize=9, fontweight="bold")
    axes[i].set_xticklabels(["Normal", "Attack"], fontsize=8)
    axes[i].set_xlabel("")

plt.suptitle("Box Plots: Feature Spread by Class (Outliers Visible)", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()

path = os.path.join(EDA_DIR, "boxplots_by_class.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Data Preprocessing
# #
# Full preprocessing pipeline:
# #
# - Categorical encoding (LabelEncoder fitted on combined train+test)
# - StandardScaler normalization
# - Class weight computation for imbalanced data
# - LSTM-ready 3D tensor reshaping
# #
# All preprocessing objects are saved for inference use.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:49.324956Z","iopub.execute_input":"2026-05-10T16:38:49.325222Z","iopub.status.idle":"2026-05-10T16:38:49.353434Z","shell.execute_reply.started":"2026-05-10T16:38:49.325200Z","shell.execute_reply":"2026-05-10T16:38:49.352714Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 : PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  DROPPING UNUSED COLUMNS")
print("=" * 70)

drop_columns = ["attack", "difficulty"]
train_df.drop(columns=drop_columns, inplace=True)
test_df.drop(columns=drop_columns,  inplace=True)

print("Columns Dropped    : attack, difficulty")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:49.354385Z","iopub.execute_input":"2026-05-10T16:38:49.354788Z","iopub.status.idle":"2026-05-10T16:38:49.444375Z","shell.execute_reply.started":"2026-05-10T16:38:49.354763Z","shell.execute_reply":"2026-05-10T16:38:49.443683Z"}}
print("=" * 70)
print("  ENCODING CATEGORICAL FEATURES")
print("=" * 70)

categorical_columns = ["protocol_type", "service", "flag"]
encoders = {}

for column in tqdm(categorical_columns, desc="Encoding"):
    encoder = LabelEncoder()
    combined_data = pd.concat([train_df[column], test_df[column]])
    encoder.fit(combined_data)
    train_df[column] = encoder.transform(train_df[column])
    test_df[column]  = encoder.transform(test_df[column])
    encoders[column] = encoder
    print(f"  {column:<20} → {len(encoder.classes_)} unique classes")

MASTER_LOG["preprocessing"]["categorical_columns"] = categorical_columns
MASTER_LOG["preprocessing"]["encoding"]            = "LabelEncoder (fit on combined train+test)"
print("Categorical Encoding Completed")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:49.445754Z","iopub.execute_input":"2026-05-10T16:38:49.446092Z","iopub.status.idle":"2026-05-10T16:38:49.474381Z","shell.execute_reply.started":"2026-05-10T16:38:49.446071Z","shell.execute_reply":"2026-05-10T16:38:49.473497Z"}}
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test  = test_df.drop("label", axis=1)
y_test  = test_df["label"]

print(f"X_train Shape      : {X_train.shape}")
print(f"X_test  Shape      : {X_test.shape}")
print(f"y_train Value Counts:\n{y_train.value_counts()}")
print(f"y_test  Value Counts:\n{y_test.value_counts()}")

MASTER_LOG["dataset"]["train_normal"]  = int((y_train == 0).sum())
MASTER_LOG["dataset"]["train_attack"]  = int((y_train == 1).sum())
MASTER_LOG["dataset"]["test_normal"]   = int((y_test  == 0).sum())
MASTER_LOG["dataset"]["test_attack"]   = int((y_test  == 1).sum())
MASTER_LOG["dataset"]["class_balance_train"] = round(float(y_train.mean()), 4)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:49.475518Z","iopub.execute_input":"2026-05-10T16:38:49.475882Z","iopub.status.idle":"2026-05-10T16:38:49.494744Z","shell.execute_reply.started":"2026-05-10T16:38:49.475848Z","shell.execute_reply":"2026-05-10T16:38:49.494018Z"}}
print("=" * 70)
print("  CLASS WEIGHT COMPUTATION")
print("=" * 70)

classes      = np.array([0, 1])
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}

print(f"Class Weight [0 - Normal] : {class_weight_dict[0]:.4f}")
print(f"Class Weight [1 - Attack] : {class_weight_dict[1]:.4f}")
MASTER_LOG["preprocessing"]["class_weights"] = {str(k): round(v, 4) for k, v in class_weight_dict.items()}

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:49.496593Z","iopub.execute_input":"2026-05-10T16:38:49.497023Z","iopub.status.idle":"2026-05-10T16:38:49.606544Z","shell.execute_reply.started":"2026-05-10T16:38:49.496987Z","shell.execute_reply":"2026-05-10T16:38:49.605899Z"}}
print("=" * 70)
print("  FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

MASTER_LOG["preprocessing"]["scaling"] = "StandardScaler (fit on train, transform test)"
print("Scaling Completed")

feature_names = X_train.columns.tolist()
MASTER_LOG["preprocessing"]["n_features"] = len(feature_names)
MASTER_LOG["preprocessing"]["feature_names"] = feature_names

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:49.607671Z","iopub.execute_input":"2026-05-10T16:38:49.607974Z","iopub.status.idle":"2026-05-10T16:38:49.615710Z","shell.execute_reply.started":"2026-05-10T16:38:49.607951Z","shell.execute_reply":"2026-05-10T16:38:49.614907Z"}}
print("=" * 70)
print("  RESHAPING FOR LSTM INPUT")
print("=" * 70)

X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_lstm  = X_test_scaled.reshape(X_test_scaled.shape[0],  X_test_scaled.shape[1],  1)

print(f"Train LSTM Shape   : {X_train_lstm.shape}")
print(f"Test  LSTM Shape   : {X_test_lstm.shape}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Feature Importance Analysis
# #
# A Random Forest classifier is used to quickly estimate feature importances
# before deep learning training. This helps understand which features the
# dataset relies on and validates the model's behaviour.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:49.617398Z","iopub.execute_input":"2026-05-10T16:38:49.617925Z","iopub.status.idle":"2026-05-10T16:38:52.390156Z","shell.execute_reply.started":"2026-05-10T16:38:49.617880Z","shell.execute_reply":"2026-05-10T16:38:52.389482Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 : FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

print("Training Random Forest for feature importance...")
rf_model.fit(X_train_scaled, y_train)

importances = pd.Series(rf_model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

importance_df = pd.DataFrame({
    "Feature"   : importances.index,
    "Importance": importances.values.round(6)
})

importance_path = os.path.join(REPORT_DIR, "feature_importance.csv")
importance_df.to_csv(importance_path, index=False)
MASTER_LOG["artifacts"].append(importance_path)

display(importance_df.head(20))
print(f"Feature Importance Saved : {importance_path}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:52.391171Z","iopub.execute_input":"2026-05-10T16:38:52.391497Z","iopub.status.idle":"2026-05-10T16:38:53.619444Z","shell.execute_reply.started":"2026-05-10T16:38:52.391461Z","shell.execute_reply":"2026-05-10T16:38:53.618870Z"}}
# ── Figure 7 : Feature Importance Plot ───────────────────────────────────

top_n = 25
top_features = importances.head(top_n)

fig, ax = plt.subplots(figsize=(12, 9))

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
bars = ax.barh(
    range(top_n),
    top_features.values,
    color=colors,
    edgecolor="white",
    linewidth=0.5
)

ax.set_yticks(range(top_n))
ax.set_yticklabels(
    [f.replace("_", " ").title() for f in top_features.index],
    fontsize=9
)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance Score", fontsize=11)
ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
             fontsize=14, fontweight="bold", pad=12)

for bar, val in zip(bars, top_features.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=7.5)

plt.tight_layout()
path = os.path.join(EDA_DIR, "feature_importance.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Dimensionality Reduction Visualizations
# #
# PCA and t-SNE are used to visualize the high-dimensional feature space
# in 2D, colour-coded by class label.
# #
# These plots reveal how separable normal and attack traffic is in the
# latent representation space — a key indicator of model feasibility.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:53.620282Z","iopub.execute_input":"2026-05-10T16:38:53.620643Z","iopub.status.idle":"2026-05-10T16:38:57.616002Z","shell.execute_reply.started":"2026-05-10T16:38:53.620619Z","shell.execute_reply":"2026-05-10T16:38:57.615178Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 : PCA + t-SNE VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  PCA DIMENSIONALITY REDUCTION")
print("=" * 70)

pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_train_scaled)

explained = pca.explained_variance_ratio_
print(f"PCA Component 1 Explained Variance : {explained[0]*100:.2f}%")
print(f"PCA Component 2 Explained Variance : {explained[1]*100:.2f}%")
print(f"Total Explained Variance           : {sum(explained)*100:.2f}%")

MASTER_LOG["preprocessing"]["pca_variance_pc1"] = round(float(explained[0]), 4)
MASTER_LOG["preprocessing"]["pca_variance_pc2"] = round(float(explained[1]), 4)

pca_df = pd.DataFrame({
    "PC1"  : X_pca[:, 0],
    "PC2"  : X_pca[:, 1],
    "Label": y_train.values
})

fig, ax = plt.subplots(figsize=(10, 8))

for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Attack")]:
    subset = pca_df[pca_df["Label"] == label]
    ax.scatter(
        subset["PC1"], subset["PC2"],
        c=color, label=name, alpha=0.4, s=8, edgecolors="none"
    )

ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", fontsize=11)
ax.set_title("PCA — 2D Projection of NSL-KDD Features", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, markerscale=3)

plt.tight_layout()
path = os.path.join(EDA_DIR, "pca_2d.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:38:57.617088Z","iopub.execute_input":"2026-05-10T16:38:57.617395Z","iopub.status.idle":"2026-05-10T16:39:22.087001Z","shell.execute_reply.started":"2026-05-10T16:38:57.617363Z","shell.execute_reply":"2026-05-10T16:39:22.086071Z"}}
print("=" * 70)
print("  t-SNE DIMENSIONALITY REDUCTION")
print("=" * 70)

# Use stratified subsample for speed
tsne_sample = 5000
indices_0 = np.where(y_train == 0)[0]
indices_1 = np.where(y_train == 1)[0]

sample_0 = np.random.choice(indices_0, min(tsne_sample // 2, len(indices_0)), replace=False)
sample_1 = np.random.choice(indices_1, min(tsne_sample // 2, len(indices_1)), replace=False)
tsne_idx  = np.concatenate([sample_0, sample_1])
np.random.shuffle(tsne_idx)

X_tsne_in = X_train_scaled[tsne_idx]
y_tsne_in = y_train.values[tsne_idx]

print(f"t-SNE Sample Size  : {len(tsne_idx):,} (stratified)")

tsne = TSNE(
    n_components=2,
    perplexity=40,
    n_iter=1000,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

print("Running t-SNE (this may take a moment)...")
X_tsne = tsne.fit_transform(X_tsne_in)

fig, ax = plt.subplots(figsize=(10, 8))

for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Attack")]:
    mask = y_tsne_in == label
    ax.scatter(
        X_tsne[mask, 0], X_tsne[mask, 1],
        c=color, label=name, alpha=0.45, s=10, edgecolors="none"
    )

ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax.set_title(f"t-SNE — 2D Projection (n={len(tsne_idx):,} samples)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, markerscale=3)

plt.tight_layout()
path = os.path.join(EDA_DIR, "tsne_2d.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Deep Learning Model — Stacked Bidirectional LSTM
# #
# Architecture:
# #
# - Bidirectional LSTM (processes sequence in both directions)
# - Stacked LSTM layers with residual-style dropout
# - Batch normalization after each LSTM block
# - L2 weight regularization to prevent overfitting
# - Sigmoid output with binary cross-entropy loss
# - Adam optimizer with gradient clipping
# #
# Overfitting prevention:
# #
# - Dropout regularization (0.3 / 0.2)
# - L2 kernel regularization
# - Early stopping (patience = 7)
# - ReduceLROnPlateau
# - Class-weighted training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:39:22.087980Z","iopub.execute_input":"2026-05-10T16:39:22.088289Z","iopub.status.idle":"2026-05-10T16:39:23.645534Z","shell.execute_reply.started":"2026-05-10T16:39:22.088264Z","shell.execute_reply":"2026-05-10T16:39:23.644752Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 : MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  BUILDING STACKED BIDIRECTIONAL LSTM MODEL")
print("=" * 70)

N_FEATURES = X_train_lstm.shape[1]
L2_REG     = 1e-4

model = Sequential([

    # Block 1 — Bidirectional LSTM
    Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=l2(L2_REG)),
        input_shape=(N_FEATURES, 1)
    ),
    BatchNormalization(),
    Dropout(0.35),

    # Block 2 — LSTM
    LSTM(96, return_sequences=True, kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.30),

    # Block 3 — LSTM
    LSTM(64, return_sequences=False, kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.25),

    # Dense head
    Dense(128, activation="relu", kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.20),

    Dense(64, activation="relu", kernel_regularizer=l2(L2_REG)),
    Dropout(0.15),

    Dense(1, activation="sigmoid", dtype="float32"),  # float32 output for mixed precision
])

optimizer = Adam(
    learning_rate=1e-3,
    clipnorm=1.0         # gradient clipping prevents exploding gradients
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

model.summary()

total_params = model.count_params()
print(f"\nTotal Parameters   : {total_params:,}")
MASTER_LOG["model"]["architecture"]    = "Bidirectional LSTM (3 blocks) + Dense head"
MASTER_LOG["model"]["total_params"]    = total_params
MASTER_LOG["model"]["l2_reg"]          = L2_REG
MASTER_LOG["model"]["dropout_rates"]   = [0.35, 0.30, 0.25, 0.20, 0.15]
MASTER_LOG["model"]["optimizer"]       = "Adam (lr=1e-3, clipnorm=1.0)"
MASTER_LOG["model"]["loss"]            = "binary_crossentropy"

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Model Training
# #
# Training configuration:
# #
# - EarlyStopping with patience=7 (monitors val_loss)
# - ReduceLROnPlateau halves LR if val_loss stalls for 3 epochs
# - ModelCheckpoint saves the best model automatically
# - CSVLogger records per-epoch metrics to a CSV file
# - Validation split = 20%
# - Batch size = 256 (GPU-optimised)
# - Class weights applied for imbalanced data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:39:23.646613Z","iopub.execute_input":"2026-05-10T16:39:23.646930Z","iopub.status.idle":"2026-05-10T16:39:23.653122Z","shell.execute_reply.started":"2026-05-10T16:39:23.646905Z","shell.execute_reply":"2026-05-10T16:39:23.652491Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 : CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

best_model_path = os.path.join(MODEL_DIR, "best_model_checkpoint.h5")
epoch_log_path  = os.path.join(LOG_DIR,   "training_epoch_log.csv")

early_stopping = EarlyStopping(
    monitor="val_auc",
    patience=7,
    restore_best_weights=True,
    verbose=1,
    mode="max"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_auc",
    save_best_only=True,
    mode="max",
    verbose=1
)

csv_logger = CSVLogger(
    epoch_log_path,
    separator=",",
    append=False
)

print("Callbacks Initialized:")
print(f"  EarlyStopping      → monitor=val_auc, patience=7")
print(f"  ReduceLROnPlateau  → factor=0.5, patience=3, min_lr=1e-6")
print(f"  ModelCheckpoint    → {best_model_path}")
print(f"  CSVLogger          → {epoch_log_path}")
MASTER_LOG["artifacts"].append(epoch_log_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:39:23.654061Z","iopub.execute_input":"2026-05-10T16:39:23.654395Z","iopub.status.idle":"2026-05-10T16:41:47.950761Z","shell.execute_reply.started":"2026-05-10T16:39:23.654349Z","shell.execute_reply":"2026-05-10T16:41:47.950153Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 : MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  TRAINING MODEL")
print("=" * 70)

EPOCHS     = 50
BATCH_SIZE = 256

train_start = time()

history = model.fit(
    X_train_lstm,
    y_train,
    validation_split=0.20,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, model_checkpoint, csv_logger],
    verbose=1
)

train_end   = time()
train_time  = round(train_end - train_start, 2)

epochs_run  = len(history.history["loss"])
best_val_auc = max(history.history["val_auc"])

print(f"\nTraining Completed")
print(f"Epochs Run         : {epochs_run} / {EPOCHS}")
print(f"Training Time      : {train_time:.1f}s")
print(f"Best Val AUC       : {best_val_auc:.4f}")

MASTER_LOG["training"]["epochs_run"]     = epochs_run
MASTER_LOG["training"]["training_time_s"]= train_time
MASTER_LOG["training"]["best_val_auc"]   = round(float(best_val_auc), 4)
MASTER_LOG["training"]["batch_size"]     = BATCH_SIZE
MASTER_LOG["training"]["validation_split"] = 0.20
MASTER_LOG["training"]["class_weights_used"] = True

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Training Performance Visualization
# #
# Comprehensive training curve analysis including:
# #
# - Accuracy, Loss, AUC, Precision, Recall per epoch
# - Train vs Validation gap analysis (overfitting indicator)
# - Learning rate schedule over epochs
# - Per-epoch metrics saved from CSVLogger

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:47.951901Z","iopub.execute_input":"2026-05-10T16:41:47.952370Z","iopub.status.idle":"2026-05-10T16:41:47.969246Z","shell.execute_reply.started":"2026-05-10T16:41:47.952345Z","shell.execute_reply":"2026-05-10T16:41:47.968546Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 : TRAINING VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

epoch_df = pd.read_csv(epoch_log_path)
print("Per-Epoch Metrics Log:")
display(epoch_df.tail(10))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:47.970191Z","iopub.execute_input":"2026-05-10T16:41:47.970679Z","iopub.status.idle":"2026-05-10T16:41:51.183416Z","shell.execute_reply.started":"2026-05-10T16:41:47.970646Z","shell.execute_reply":"2026-05-10T16:41:51.182697Z"}}
# ── Figure 8 : Comprehensive Training Dashboard ───────────────────────────

fig = plt.figure(figsize=(22, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

metric_pairs = [
    ("accuracy",  "val_accuracy",  "Accuracy",  "royalblue",  "tomato"),
    ("loss",      "val_loss",      "Loss",      "darkorange", "purple"),
    ("auc",       "val_auc",       "AUC",       "seagreen",   "crimson"),
    ("precision", "val_precision", "Precision", "steelblue",  "darkorange"),
    ("recall",    "val_recall",    "Recall",    "mediumvioletred", "teal"),
]

for idx, (train_key, val_key, title, c_train, c_val) in enumerate(metric_pairs):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    if train_key in history.history:
        epochs_x = range(1, epochs_run + 1)
        ax.plot(epochs_x, history.history[train_key], label=f"Train {title}",
                color=c_train, linewidth=2.0)
        ax.plot(epochs_x, history.history[val_key],   label=f"Val {title}",
                color=c_val, linewidth=2.0, linestyle="--")

        # Shade gap between train/val (overfitting zone)
        ax.fill_between(
            epochs_x,
            history.history[train_key],
            history.history[val_key],
            alpha=0.12,
            color="gray",
            label="Train-Val Gap"
        )

        # Mark best epoch
        if "loss" in train_key:
            best_ep = np.argmin(history.history[val_key]) + 1
        else:
            best_ep = np.argmax(history.history[val_key]) + 1

        ax.axvline(x=best_ep, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.annotate(f"Best\nEp.{best_ep}", xy=(best_ep, ax.get_ylim()[0]),
                    fontsize=7.5, ha="center", color="black", alpha=0.7)

        ax.set_title(f"Model {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

# Learning Rate subplot
ax_lr = fig.add_subplot(gs[1, 2])
if "lr" in epoch_df.columns:
    ax_lr.plot(epoch_df["epoch"] + 1, epoch_df["lr"],
               color="darkorange", linewidth=2.0, marker="o", markersize=4)
    ax_lr.set_yscale("log")
    ax_lr.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    ax_lr.set_xlabel("Epoch", fontsize=10)
    ax_lr.set_ylabel("Learning Rate (log scale)", fontsize=10)
    ax_lr.grid(True, alpha=0.4)

plt.suptitle("Training Dashboard — NSL-KDD LSTM Model", fontsize=18, fontweight="bold", y=1.01)

path = os.path.join(TRAIN_DIR, "training_dashboard.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:51.184287Z","iopub.execute_input":"2026-05-10T16:41:51.184574Z","iopub.status.idle":"2026-05-10T16:41:52.215557Z","shell.execute_reply.started":"2026-05-10T16:41:51.184546Z","shell.execute_reply":"2026-05-10T16:41:52.214883Z"}}
# ── Figure 9 : Overfitting Analysis ──────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Train-Val Accuracy gap
acc_gap = np.array(history.history["accuracy"]) - np.array(history.history["val_accuracy"])
axes[0].fill_between(
    range(1, epochs_run + 1), acc_gap, 0,
    where=(acc_gap > 0), alpha=0.5, color="#e74c3c", label="Overfit Zone (Train > Val)"
)
axes[0].fill_between(
    range(1, epochs_run + 1), acc_gap, 0,
    where=(acc_gap <= 0), alpha=0.5, color="#2ecc71", label="Underfit Zone (Val > Train)"
)
axes[0].axhline(y=0, color="black", linewidth=1.2, linestyle="--")
axes[0].set_title("Accuracy Gap (Train − Validation)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy Gap")
axes[0].legend(fontsize=9)

# Train-Val Loss gap
loss_gap = np.array(history.history["val_loss"]) - np.array(history.history["loss"])
axes[1].fill_between(
    range(1, epochs_run + 1), loss_gap, 0,
    where=(loss_gap > 0), alpha=0.5, color="#e74c3c", label="Overfit Zone (Val Loss > Train)"
)
axes[1].fill_between(
    range(1, epochs_run + 1), loss_gap, 0,
    where=(loss_gap <= 0), alpha=0.5, color="#2ecc71", label="Underfit Zone"
)
axes[1].axhline(y=0, color="black", linewidth=1.2, linestyle="--")
axes[1].set_title("Loss Gap (Validation − Training)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss Gap")
axes[1].legend(fontsize=9)

plt.suptitle("Overfitting / Underfitting Analysis", fontsize=15, fontweight="bold")
plt.tight_layout()

path = os.path.join(TRAIN_DIR, "overfitting_analysis.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Model Evaluation
# #
# Comprehensive evaluation using:
# #
# - Accuracy, Precision, Recall, F1-Score
# - Matthews Correlation Coefficient (MCC)
# - Cohen's Kappa
# - AUC-ROC
# - AUC-PR (Average Precision)
# - Confusion Matrix (raw + normalized)
# - ROC Curve
# - Precision-Recall Curve
# - Threshold analysis
# - Per-class classification report

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:52.216382Z","iopub.execute_input":"2026-05-10T16:41:52.216688Z","iopub.status.idle":"2026-05-10T16:41:52.891484Z","shell.execute_reply.started":"2026-05-10T16:41:52.216644Z","shell.execute_reply":"2026-05-10T16:41:52.890874Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 : MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  EVALUATING MODEL ON TEST SET")
print("=" * 70)

eval_results = model.evaluate(X_test_lstm, y_test, verbose=1, batch_size=512)
eval_names   = model.metrics_names

print("\nTest Set Metrics:")
for name, val in zip(eval_names, eval_results):
    print(f"  {name:<20} : {val:.4f}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:52.892276Z","iopub.execute_input":"2026-05-10T16:41:52.892550Z","iopub.status.idle":"2026-05-10T16:41:54.048317Z","shell.execute_reply.started":"2026-05-10T16:41:52.892527Z","shell.execute_reply":"2026-05-10T16:41:54.047641Z"}}
print("=" * 70)
print("  GENERATING PREDICTIONS")
print("=" * 70)

y_pred_prob = model.predict(X_test_lstm, batch_size=512, verbose=1).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("Predictions Completed")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:54.049364Z","iopub.execute_input":"2026-05-10T16:41:54.049696Z","iopub.status.idle":"2026-05-10T16:41:54.093193Z","shell.execute_reply.started":"2026-05-10T16:41:54.049670Z","shell.execute_reply":"2026-05-10T16:41:54.092572Z"}}
# ── Full Metrics Suite ────────────────────────────────────────────────────

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
mcc       = matthews_corrcoef(y_test, y_pred)
kappa     = cohen_kappa_score(y_test, y_pred)

fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_prob)
auc_roc   = auc(fpr_arr, tpr_arr)
auc_pr    = average_precision_score(y_test, y_pred_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
fpr_metric  = fp / (fp + tn)

metrics_dict = {
    "Accuracy"         : round(accuracy,    4),
    "Precision"        : round(precision,   4),
    "Recall (TPR)"     : round(recall,      4),
    "Specificity (TNR)": round(specificity, 4),
    "F1 Score"         : round(f1,          4),
    "MCC"              : round(mcc,         4),
    "Cohen Kappa"      : round(kappa,       4),
    "AUC-ROC"          : round(auc_roc,     4),
    "AUC-PR"           : round(auc_pr,      4),
    "False Positive Rate": round(fpr_metric,4),
    "Negative Predictive Value": round(npv, 4),
    "True Positives"   : int(tp),
    "True Negatives"   : int(tn),
    "False Positives"  : int(fp),
    "False Negatives"  : int(fn),
}

metrics_df = pd.DataFrame(
    metrics_dict.items(),
    columns=["Metric", "Value"]
)

display(metrics_df)

metrics_csv_path = os.path.join(REPORT_DIR, "evaluation_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Metrics Saved      : {metrics_csv_path}")
MASTER_LOG["evaluation"] = metrics_dict
MASTER_LOG["artifacts"].append(metrics_csv_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:54.094152Z","iopub.execute_input":"2026-05-10T16:41:54.094491Z","iopub.status.idle":"2026-05-10T16:41:54.113696Z","shell.execute_reply.started":"2026-05-10T16:41:54.094468Z","shell.execute_reply":"2026-05-10T16:41:54.113059Z"}}
print("=" * 70)
print("  CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"])
print(report)

report_path = os.path.join(REPORT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"NSL-KDD LSTM Intrusion Detection System\n")
    f.write(f"Run ID: {RUN_ID}\n")
    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
    f.write("=" * 60 + "\n")
    f.write(report)

print(f"Report Saved       : {report_path}")
MASTER_LOG["artifacts"].append(report_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:54.114746Z","iopub.execute_input":"2026-05-10T16:41:54.115178Z","iopub.status.idle":"2026-05-10T16:41:55.102579Z","shell.execute_reply.started":"2026-05-10T16:41:54.115143Z","shell.execute_reply":"2026-05-10T16:41:55.101769Z"}}
# ── Figure 10 : Confusion Matrix (Raw + Normalized) ──────────────────────

cm_raw  = confusion_matrix(y_test, y_pred)
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw confusion matrix
sns.heatmap(
    cm_raw, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"],
    ax=axes[0],
    linewidths=1,
    linecolor="white",
    annot_kws={"size": 16, "fontweight": "bold"}
)
axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Predicted Label", fontsize=11)
axes[0].set_ylabel("True Label", fontsize=11)

# Normalized confusion matrix
sns.heatmap(
    cm_norm, annot=True, fmt=".3f", cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"],
    ax=axes[1],
    linewidths=1,
    linecolor="white",
    annot_kws={"size": 16, "fontweight": "bold"},
    vmin=0, vmax=1
)
axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Predicted Label", fontsize=11)
axes[1].set_ylabel("True Label", fontsize=11)

plt.suptitle("Confusion Matrix Analysis", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(EVAL_DIR, "confusion_matrix.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:55.103527Z","iopub.execute_input":"2026-05-10T16:41:55.103926Z","iopub.status.idle":"2026-05-10T16:41:55.964772Z","shell.execute_reply.started":"2026-05-10T16:41:55.103901Z","shell.execute_reply":"2026-05-10T16:41:55.963974Z"}}
# ── Figure 11 : ROC Curve ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(fpr_arr, tpr_arr, color="royalblue", linewidth=2.5, label=f"LSTM Model (AUC = {auc_roc:.4f})")
ax.fill_between(fpr_arr, tpr_arr, alpha=0.12, color="royalblue")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier (AUC = 0.5)")
ax.plot([0, 0, 1], [0, 1, 1], "g:", linewidth=1.5, label="Perfect Classifier (AUC = 1.0)")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curve — Intrusion Detection", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.4)

# Operating point at 0.5 threshold
ax.scatter(fpr_metric, recall, color="red", s=100, zorder=5, label="Operating Point (0.5)")
ax.annotate(
    f"  Op. Point\n  FPR={fpr_metric:.3f}\n  TPR={recall:.3f}",
    xy=(fpr_metric, recall), fontsize=9, color="red"
)

plt.tight_layout()
path = os.path.join(EVAL_DIR, "roc_curve.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:55.966031Z","iopub.execute_input":"2026-05-10T16:41:55.966677Z","iopub.status.idle":"2026-05-10T16:41:57.489304Z","shell.execute_reply.started":"2026-05-10T16:41:55.966651Z","shell.execute_reply":"2026-05-10T16:41:57.488521Z"}}
# ── Figure 12 : Precision-Recall Curve ───────────────────────────────────

prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_pred_prob)
baseline = y_test.mean()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# PR Curve
axes[0].plot(rec_arr, prec_arr, color="darkorange", linewidth=2.5,
             label=f"LSTM (AUC-PR = {auc_pr:.4f})")
axes[0].fill_between(rec_arr, prec_arr, alpha=0.12, color="darkorange")
axes[0].axhline(y=baseline, color="navy", linestyle="--", linewidth=1.5,
                label=f"Baseline (attack rate = {baseline:.3f})")
axes[0].set_xlabel("Recall", fontsize=12)
axes[0].set_ylabel("Precision", fontsize=12)
axes[0].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=11)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.02])
axes[0].grid(True, alpha=0.4)

# Threshold analysis
thresh_plot = thresh_arr[::max(1, len(thresh_arr)//200)]
prec_plot   = prec_arr[::max(1, len(thresh_arr)//200)]
rec_plot    = rec_arr[::max(1, len(thresh_arr)//200)]

axes[1].plot(thresh_plot, prec_plot[:-1] if len(prec_plot) > len(thresh_plot) else prec_plot,
             color="steelblue", linewidth=2, label="Precision")
axes[1].plot(thresh_plot, rec_plot[:-1]  if len(rec_plot)  > len(thresh_plot) else rec_plot,
             color="crimson", linewidth=2, label="Recall")
axes[1].axvline(x=0.5, color="black", linestyle=":", linewidth=1.5, label="Default Threshold (0.5)")
axes[1].set_xlabel("Classification Threshold", fontsize=12)
axes[1].set_ylabel("Score", fontsize=12)
axes[1].set_title("Precision & Recall vs Threshold", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])
axes[1].grid(True, alpha=0.4)

plt.suptitle("Precision-Recall Analysis", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(EVAL_DIR, "precision_recall_curve.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:57.490380Z","iopub.execute_input":"2026-05-10T16:41:57.490691Z","iopub.status.idle":"2026-05-10T16:41:58.828191Z","shell.execute_reply.started":"2026-05-10T16:41:57.490660Z","shell.execute_reply":"2026-05-10T16:41:58.827321Z"}}
# ── Figure 13 : Prediction Probability Distribution ──────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# By class
for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Attack")]:
    mask = y_test == label
    axes[0].hist(
        y_pred_prob[mask], bins=60, alpha=0.65, density=True,
        color=color, label=f"{name} (n={mask.sum():,})", edgecolor="white"
    )

axes[0].axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Boundary")
axes[0].set_xlabel("Predicted Probability (Attack)", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title("Prediction Probability Distribution by True Class", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)

# Calibration-style scatter
axes[1].hist(y_pred_prob, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.5")
axes[1].set_xlabel("Predicted Probability", fontsize=11)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title("Overall Prediction Probability Histogram", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)

plt.suptitle("Model Confidence Analysis", fontsize=15, fontweight="bold")
plt.tight_layout()

path = os.path.join(EVAL_DIR, "prediction_probability_distribution.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:58.829183Z","iopub.execute_input":"2026-05-10T16:41:58.829475Z","iopub.status.idle":"2026-05-10T16:41:59.978295Z","shell.execute_reply.started":"2026-05-10T16:41:58.829441Z","shell.execute_reply":"2026-05-10T16:41:59.977603Z"}}
# ── Figure 14 : Comprehensive Metrics Dashboard ───────────────────────────

score_labels  = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "AUC-ROC", "AUC-PR", "MCC*"]
score_values  = [accuracy, precision, recall, specificity, f1, auc_roc, auc_pr, (mcc + 1) / 2]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Radar / bar chart of metrics
colors_bar = sns.color_palette("husl", len(score_labels))
bars = axes[0].bar(score_labels, score_values[:len(score_labels)], color=colors_bar, edgecolor="white", linewidth=0.6)
axes[0].set_ylim(0, 1.08)
axes[0].set_title("Model Performance Metrics Summary\n(*MCC normalized to [0,1])",
                   fontsize=13, fontweight="bold")
axes[0].set_ylabel("Score", fontsize=11)
axes[0].tick_params(axis="x", rotation=30)

for bar, val in zip(bars, score_values):
    color = "#2ecc71" if val >= 0.9 else "#f39c12" if val >= 0.75 else "#e74c3c"
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color=color)

# Confusion matrix details table
cm_details = {
    "TP (Attack correctly detected)"   : f"{tp:,}",
    "TN (Normal correctly classified)" : f"{tn:,}",
    "FP (Normal misclassified as Attack)": f"{fp:,}",
    "FN (Attack missed)"               : f"{fn:,}",
    "Total Test Samples"               : f"{len(y_test):,}",
    "Detection Rate (Recall)"          : f"{recall:.4f}",
    "False Alarm Rate (FPR)"           : f"{fpr_metric:.4f}",
}

axes[1].axis("off")
table_data = [[k, v] for k, v in cm_details.items()]
table = axes[1].table(
    cellText=table_data,
    colLabels=["Metric", "Value"],
    loc="center",
    cellLoc="left"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.5, 2.2)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#ecf0f1")

axes[1].set_title("Detailed Classification Statistics", fontsize=13, fontweight="bold", pad=20)

plt.suptitle("Complete Evaluation Report", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(EVAL_DIR, "metrics_dashboard.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Blockchain Integrated Security Alert System
# #
# A SHA-256 based blockchain is used to store:
# #
# - All detected attack events
# - Threat level classification (low / medium / high / critical)
# - Prediction confidence score
# - Tamper-proof hash chain
# - Chain integrity validation
# #
# The blockchain ensures immutability of all security alerts.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:59.982318Z","iopub.execute_input":"2026-05-10T16:41:59.982535Z","iopub.status.idle":"2026-05-10T16:41:59.996592Z","shell.execute_reply.started":"2026-05-10T16:41:59.982516Z","shell.execute_reply":"2026-05-10T16:41:59.995635Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 : BLOCKCHAIN FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  INITIALIZING BLOCKCHAIN FRAMEWORK")
print("=" * 70)

def classify_threat_level(confidence):
    if confidence >= 0.95:
        return "CRITICAL"
    elif confidence >= 0.85:
        return "HIGH"
    elif confidence >= 0.70:
        return "MEDIUM"
    else:
        return "LOW"


class Block:

    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        self.index         = index
        self.timestamp     = timestamp
        self.data          = data
        self.previous_hash = previous_hash
        self.nonce         = nonce
        self.hash          = self.calculate_hash()

    def calculate_hash(self):
        block_content = json.dumps({
            "index"        : self.index,
            "timestamp"    : self.timestamp,
            "data"         : self.data,
            "previous_hash": self.previous_hash,
            "nonce"        : self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_content).hexdigest()

    def to_dict(self):
        return {
            "index"        : self.index,
            "timestamp"    : self.timestamp,
            "data"         : self.data,
            "previous_hash": self.previous_hash,
            "nonce"        : self.nonce,
            "hash"         : self.hash
        }


class Blockchain:

    def __init__(self, chain_id="IDS-CHAIN-01"):
        self.chain_id  = chain_id
        self.chain     = [self._create_genesis_block()]
        self.alerts    = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    def _create_genesis_block(self):
        return Block(
            index=0,
            timestamp=time(),
            data={
                "type"       : "GENESIS",
                "system"     : "NSL-KDD IDS Framework",
                "version"    : "2.0",
                "description": "Blockchain Intrusion Detection Alert Log"
            },
            previous_hash="0000000000000000"
        )

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        new_block = Block(
            index=len(self.chain),
            timestamp=time(),
            data=data,
            previous_hash=self.get_latest_block().hash
        )
        self.chain.append(new_block)
        if "threat_level" in data:
            self.alerts[data["threat_level"]] += 1
        return new_block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.hash != curr.calculate_hash():
                return False, f"Block {i} hash mismatch"
            if curr.previous_hash != prev.hash:
                return False, f"Block {i} broken chain link"
        return True, "Chain is valid"

    def get_stats(self):
        return {
            "chain_id"    : self.chain_id,
            "total_blocks": len(self.chain),
            "alert_counts": self.alerts,
            "is_valid"    : self.is_chain_valid()[0]
        }

print("Blockchain Framework Ready")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:41:59.997594Z","iopub.execute_input":"2026-05-10T16:41:59.998028Z","iopub.status.idle":"2026-05-10T16:42:00.046295Z","shell.execute_reply.started":"2026-05-10T16:41:59.997996Z","shell.execute_reply":"2026-05-10T16:42:00.045589Z"}}
print("=" * 70)
print("  LOGGING ATTACK ALERTS TO BLOCKCHAIN")
print("=" * 70)

blockchain = Blockchain(chain_id=f"IDS-{RUN_ID}")

attack_indices    = np.where(y_pred == 1)[0]
total_attacks     = len(attack_indices)
max_blocks        = 300
sample_indices    = attack_indices[:max_blocks]

print(f"Total Detected Attacks : {total_attacks:,}")
print(f"Logging to Blockchain  : {len(sample_indices):,} records")

for idx in tqdm(sample_indices, desc="Writing Blockchain"):
    confidence   = float(y_pred_prob[idx])
    threat_level = classify_threat_level(confidence)

    alert_record = {
        "type"            : "ATTACK_ALERT",
        "record_index"    : int(idx),
        "confidence_score": round(confidence, 6),
        "threat_level"    : threat_level,
        "prediction"      : "Attack",
        "true_label"      : "Attack" if int(y_test.iloc[idx]) == 1 else "Normal",
        "is_true_positive": int(y_test.iloc[idx]) == 1,
        "timestamp"       : datetime.datetime.now().isoformat(),
        "system"          : "NSL-KDD LSTM IDS"
    }

    blockchain.add_block(alert_record)

valid, msg = blockchain.is_chain_valid()
stats      = blockchain.get_stats()

print(f"\nBlockchain Integrity   : {msg}")
print(f"Total Blocks           : {stats['total_blocks']}")
print(f"Alert Breakdown:")
for level, count in stats["alert_counts"].items():
    print(f"  {level:<12} : {count}")

MASTER_LOG["blockchain"] = stats

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:42:00.047368Z","iopub.execute_input":"2026-05-10T16:42:00.047653Z","iopub.status.idle":"2026-05-10T16:42:00.873561Z","shell.execute_reply.started":"2026-05-10T16:42:00.047623Z","shell.execute_reply":"2026-05-10T16:42:00.872695Z"}}
# ── Figure 15 : Blockchain Alert Distribution ─────────────────────────────

alert_counts = stats["alert_counts"]
alert_labels = list(alert_counts.keys())
alert_values = list(alert_counts.values())
alert_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]  # CRITICAL → LOW

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
wedges, texts, autotexts = axes[0].pie(
    alert_values,
    labels=alert_labels,
    autopct="%1.1f%%",
    colors=alert_colors,
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for autotext in autotexts:
    autotext.set_fontweight("bold")
    autotext.set_fontsize(11)
axes[0].set_title("Threat Level Distribution", fontsize=14, fontweight="bold")

# Bar chart
bars = axes[1].bar(alert_labels, alert_values, color=alert_colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, alert_values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", fontsize=11, fontweight="bold")
axes[1].set_title("Threat Level Count", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Threat Level")
axes[1].set_ylabel("Count")

plt.suptitle("Blockchain Alert Analysis", fontsize=16, fontweight="bold")
plt.tight_layout()

path = os.path.join(EVAL_DIR, "blockchain_alert_distribution.png")
plt.savefig(path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Figure Saved       : {path}")
MASTER_LOG["artifacts"].append(path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:42:00.875141Z","iopub.execute_input":"2026-05-10T16:42:00.875470Z","iopub.status.idle":"2026-05-10T16:42:00.902048Z","shell.execute_reply.started":"2026-05-10T16:42:00.875446Z","shell.execute_reply":"2026-05-10T16:42:00.901075Z"}}
# Save blockchain records to CSV + JSON

blockchain_records = [block.to_dict() for block in blockchain.chain]
blockchain_df      = pd.DataFrame(blockchain_records)

blockchain_df["data"] = blockchain_df["data"].astype(str)

blockchain_csv_path  = os.path.join(BLOCKCHAIN_DIR, "blockchain_logs.csv")
blockchain_json_path = os.path.join(BLOCKCHAIN_DIR, "blockchain_logs.json")

blockchain_df.to_csv(blockchain_csv_path, index=False)

with open(blockchain_json_path, "w") as f:
    json.dump(blockchain_records, f, indent=4, default=str)

print(f"Blockchain CSV Saved : {blockchain_csv_path}")
print(f"Blockchain JSON Saved: {blockchain_json_path}")

display(blockchain_df.head())

MASTER_LOG["artifacts"].append(blockchain_csv_path)
MASTER_LOG["artifacts"].append(blockchain_json_path)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Saving All Model Artifacts
# #
# Saved outputs:
# #
# - Trained LSTM model (.h5)
# - Feature scaler (.pkl)
# - Label encoders (.pkl)
# - Feature names (.json)
# - Master run log (.json)
# - HTML project report

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:42:00.903067Z","iopub.execute_input":"2026-05-10T16:42:00.903376Z","iopub.status.idle":"2026-05-10T16:42:01.003274Z","shell.execute_reply.started":"2026-05-10T16:42:00.903352Z","shell.execute_reply":"2026-05-10T16:42:01.002372Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 : SAVING ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  SAVING MODEL & ARTIFACTS")
print("=" * 70)

# Keras model
model_path = os.path.join(MODEL_DIR, "nsl_kdd_lstm_model.h5")
model.save(model_path)
print(f"Model Saved        : {model_path}")
MASTER_LOG["artifacts"].append(model_path)

# Scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler Saved       : {scaler_path}")
MASTER_LOG["artifacts"].append(scaler_path)

# Encoders
encoders_path = os.path.join(MODEL_DIR, "encoders.pkl")
joblib.dump(encoders, encoders_path)
print(f"Encoders Saved     : {encoders_path}")
MASTER_LOG["artifacts"].append(encoders_path)

# Feature names
feature_meta = {
    "feature_names"       : feature_names,
    "categorical_features": categorical_columns,
    "n_features"          : len(feature_names),
    "input_shape"         : list(X_train_lstm.shape[1:])
}
feature_meta_path = os.path.join(MODEL_DIR, "feature_metadata.json")
with open(feature_meta_path, "w") as f:
    json.dump(feature_meta, f, indent=4)
print(f"Feature Meta Saved : {feature_meta_path}")
MASTER_LOG["artifacts"].append(feature_meta_path)

# Master log
master_log_path = os.path.join(REPORT_DIR, "master_run_log.json")
with open(master_log_path, "w") as f:
    json.dump(MASTER_LOG, f, indent=4, default=str)
print(f"Master Log Saved   : {master_log_path}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:42:01.004365Z","iopub.execute_input":"2026-05-10T16:42:01.004670Z","iopub.status.idle":"2026-05-10T16:42:01.015583Z","shell.execute_reply.started":"2026-05-10T16:42:01.004639Z","shell.execute_reply":"2026-05-10T16:42:01.015020Z"}}
# ── HTML Report Generation ────────────────────────────────────────────────

print("=" * 70)
print("  GENERATING HTML REPORT")
print("=" * 70)

html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSL-KDD IDS Report — {RUN_ID}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #f5f6fa; color: #2c3e50; }}
  .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 40px 60px; }}
  .header h1 {{ margin: 0; font-size: 28px; }}
  .header p  {{ margin: 8px 0 0; opacity: 0.85; }}
  .container {{ max-width: 1100px; margin: 30px auto; padding: 0 30px; }}
  .card {{ background: white; border-radius: 10px; padding: 25px 30px; margin-bottom: 25px;
           box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
  .card h2 {{ margin-top: 0; color: #2980b9; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #2c3e50; color: white; padding: 10px 15px; text-align: left; }}
  td {{ padding: 9px 15px; border-bottom: 1px solid #ecf0f1; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
  .badge-green  {{ background: #d4edda; color: #155724; }}
  .badge-red    {{ background: #f8d7da; color: #721c24; }}
  .badge-blue   {{ background: #cce5ff; color: #004085; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
  .metric-box {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; border-left: 4px solid #3498db; }}
  .metric-box .val {{ font-size: 26px; font-weight: bold; color: #2c3e50; }}
  .metric-box .lbl {{ font-size: 12px; color: #7f8c8d; margin-top: 4px; }}
  .footer {{ text-align: center; padding: 20px; color: #7f8c8d; font-size: 13px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ Advanced Intrusion Detection System — Project Report</h1>
  <p>NSL-KDD Dataset · Bidirectional LSTM · Blockchain Alert Logging · Run ID: {RUN_ID}</p>
  <p>Generated: {datetime.datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
</div>

<div class="container">

  <div class="card">
    <h2>📊 Dataset Summary</h2>
    <div class="metric-grid">
      <div class="metric-box"><div class="val">{MASTER_LOG['dataset']['train_samples']:,}</div><div class="lbl">Training Samples</div></div>
      <div class="metric-box"><div class="val">{MASTER_LOG['dataset']['test_samples']:,}</div><div class="lbl">Test Samples</div></div>
      <div class="metric-box"><div class="val">{MASTER_LOG['preprocessing']['n_features']}</div><div class="lbl">Features</div></div>
      <div class="metric-box"><div class="val">{MASTER_LOG['dataset']['unique_attack_types']}</div><div class="lbl">Attack Types</div></div>
      <div class="metric-box"><div class="val">{MASTER_LOG['dataset']['train_normal']:,}</div><div class="lbl">Normal Samples (Train)</div></div>
      <div class="metric-box"><div class="val">{MASTER_LOG['dataset']['train_attack']:,}</div><div class="lbl">Attack Samples (Train)</div></div>
    </div>
  </div>

  <div class="card">
    <h2>⚡ Model Performance</h2>
    <div class="metric-grid">
      <div class="metric-box"><div class="val">{metrics_dict['Accuracy']}</div><div class="lbl">Accuracy</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['Precision']}</div><div class="lbl">Precision</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['Recall (TPR)']}</div><div class="lbl">Recall (TPR)</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['F1 Score']}</div><div class="lbl">F1 Score</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['AUC-ROC']}</div><div class="lbl">AUC-ROC</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['AUC-PR']}</div><div class="lbl">AUC-PR</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['MCC']}</div><div class="lbl">MCC</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['Cohen Kappa']}</div><div class="lbl">Cohen's Kappa</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['Specificity (TNR)']}</div><div class="lbl">Specificity (TNR)</div></div>
      <div class="metric-box"><div class="val">{metrics_dict['False Positive Rate']}</div><div class="lbl">False Positive Rate</div></div>
    </div>
  </div>

  <div class="card">
    <h2>🏗️ Model Architecture</h2>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Architecture</td><td>{MASTER_LOG['model']['architecture']}</td></tr>
      <tr><td>Total Parameters</td><td>{MASTER_LOG['model']['total_params']:,}</td></tr>
      <tr><td>Optimizer</td><td>{MASTER_LOG['model']['optimizer']}</td></tr>
      <tr><td>Loss Function</td><td>{MASTER_LOG['model']['loss']}</td></tr>
      <tr><td>L2 Regularization</td><td>{MASTER_LOG['model']['l2_reg']}</td></tr>
      <tr><td>Mixed Precision</td><td>{'float16 (GPU)' if MASTER_LOG['gpu_available'] else 'Disabled (CPU)'}</td></tr>
      <tr><td>Epochs Run</td><td>{MASTER_LOG['training']['epochs_run']} / {EPOCHS}</td></tr>
      <tr><td>Training Time</td><td>{MASTER_LOG['training']['training_time_s']:.1f}s</td></tr>
      <tr><td>Best Val AUC</td><td>{MASTER_LOG['training']['best_val_auc']}</td></tr>
      <tr><td>Class Weights</td><td>Normal={MASTER_LOG['preprocessing']['class_weights']['0']}, Attack={MASTER_LOG['preprocessing']['class_weights']['1']}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>🔗 Blockchain Alert Log</h2>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Chain ID</td><td>{stats['chain_id']}</td></tr>
      <tr><td>Total Blocks</td><td>{stats['total_blocks']}</td></tr>
      <tr><td>Chain Integrity</td><td><span class="badge {'badge-green' if stats['is_valid'] else 'badge-red'}">{'✅ Valid' if stats['is_valid'] else '❌ Compromised'}</span></td></tr>
      <tr><td>CRITICAL Alerts</td><td>{stats['alert_counts']['CRITICAL']}</td></tr>
      <tr><td>HIGH Alerts</td><td>{stats['alert_counts']['HIGH']}</td></tr>
      <tr><td>MEDIUM Alerts</td><td>{stats['alert_counts']['MEDIUM']}</td></tr>
      <tr><td>LOW Alerts</td><td>{stats['alert_counts']['LOW']}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>📁 Generated Artifacts</h2>
    <table>
      <tr><th>#</th><th>File Path</th></tr>
      {''.join(f"<tr><td>{i+1}</td><td><code>{p}</code></td></tr>" for i, p in enumerate(MASTER_LOG['artifacts']))}
    </table>
  </div>

</div>
<div class="footer">
  Generated by NSL-KDD Advanced IDS Framework · Run {RUN_ID} · Powered by TensorFlow {tf.__version__}
</div>
</body>
</html>
"""

html_path = os.path.join(REPORT_DIR, f"project_report_{RUN_ID}.html")
with open(html_path, "w") as f:
    f.write(html_report)

print(f"HTML Report Saved  : {html_path}")
MASTER_LOG["artifacts"].append(html_path)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Final Project Summary
# #
# Complete summary of all computed metrics, training stats,
# blockchain data, and artifact locations.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-05-10T16:42:01.016605Z","iopub.execute_input":"2026-05-10T16:42:01.016916Z","iopub.status.idle":"2026-05-10T16:42:01.036399Z","shell.execute_reply.started":"2026-05-10T16:42:01.016894Z","shell.execute_reply":"2026-05-10T16:42:01.035625Z"}}
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 : FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  FINAL PROJECT SUMMARY")
print("=" * 70)

print(f"\n{'─'*40}")
print("  DATASET")
print(f"{'─'*40}")
print(f"  Training Samples       : {len(train_df):>10,}")
print(f"  Testing Samples        : {len(test_df):>10,}")
print(f"  Total Features         : {len(feature_names):>10}")
print(f"  Unique Attack Types    : {train_df['label'].nunique():>10}")

print(f"\n{'─'*40}")
print("  TRAINING")
print(f"{'─'*40}")
print(f"  Epochs Run             : {epochs_run:>10} / {EPOCHS}")
print(f"  Training Time          : {train_time:>9.1f}s")
print(f"  Best Val AUC           : {best_val_auc:>10.4f}")
print(f"  GPU Accelerated        : {'Yes' if len(gpus) > 0 else 'No':>10}")

print(f"\n{'─'*40}")
print("  PERFORMANCE METRICS")
print(f"{'─'*40}")
print(f"  Accuracy               : {accuracy:>10.4f}")
print(f"  Precision              : {precision:>10.4f}")
print(f"  Recall (TPR)           : {recall:>10.4f}")
print(f"  Specificity (TNR)      : {specificity:>10.4f}")
print(f"  F1 Score               : {f1:>10.4f}")
print(f"  MCC                    : {mcc:>10.4f}")
print(f"  Cohen's Kappa          : {kappa:>10.4f}")
print(f"  AUC-ROC                : {auc_roc:>10.4f}")
print(f"  AUC-PR                 : {auc_pr:>10.4f}")
print(f"  False Positive Rate    : {fpr_metric:>10.4f}")

print(f"\n{'─'*40}")
print("  CONFUSION MATRIX")
print(f"{'─'*40}")
print(f"  True Positives  (TP)   : {tp:>10,}")
print(f"  True Negatives  (TN)   : {tn:>10,}")
print(f"  False Positives (FP)   : {fp:>10,}")
print(f"  False Negatives (FN)   : {fn:>10,}")

print(f"\n{'─'*40}")
print("  BLOCKCHAIN")
print(f"{'─'*40}")
print(f"  Chain Valid            : {str(valid):>10}")
print(f"  Total Blocks           : {len(blockchain.chain):>10,}")
for level, count in stats["alert_counts"].items():
    print(f"  {level:<22} : {count:>10,}")

print(f"\n{'─'*40}")
print("  OUTPUT LOCATIONS")
print(f"{'─'*40}")
print(f"  Figures     → {FIGURE_DIR}")
print(f"  Models      → {MODEL_DIR}")
print(f"  Reports     → {REPORT_DIR}")
print(f"  Blockchain  → {BLOCKCHAIN_DIR}")
print(f"  Logs        → {LOG_DIR}")

print(f"\n{'='*70}")
print(f"  PROJECT COMPLETED SUCCESSFULLY — Run ID: {RUN_ID}")
print(f"{'='*70}")