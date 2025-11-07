# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import numpy as np
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mord  # Ordinal Logistic Regression 模型
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import plotly.express as px 
import re
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

DATA_DIR = Path("../data")

# %%
csat_df = pd.read_csv(f"{DATA_DIR}/processed_csat_data.csv", encoding="utf-8-sig")

# 欄位設定（依你實際資料；你已經有 remarks_category / response_lag_min / region）
TARGET_COL = "CSAT Score"

NUMERIC_COLS = [
    "response_lag_min",         # 你已經算好
    # "Item_price",               # 金額，缺失值多，且可以跟product cate一起看
]

CATEGORICAL_COLS = [
    "channel_name",
    "category",
    "Sub-category",
    "Tenure Bucket",
    "Agent Shift",
    # "region",             # 你已經算好
    "Product_category",
]

# 僅保留分析需要的欄位
use_cols = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]
df = csat_df[use_cols].copy()

# 建立二分類標籤：滿意 vs 不滿意
# 1~3 = 不滿意 (0)，4~5 = 滿意 (1)
df = df[pd.to_numeric(df[TARGET_COL], errors="coerce").notna()]
df[TARGET_COL] = df[TARGET_COL].astype(int).clip(1, 5)
df["satisfied_flag"] = np.where(df[TARGET_COL] >= 4, 1, 0)

# 簡單檢查缺失
print("缺失值：")
df[use_cols].isna().mean().sort_values(ascending=False).head(10)


# %% Train/Test split 與前處理（數值補中位、標準化；類別補眾數、OneHot）
X_raw = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
y = df["satisfied_flag"].copy()

# 因 response_lag_min 缺失的比例跟 y 的差不多，先刪掉看看
y = y[X_raw['response_lag_min'].notna()]
X_raw = X_raw[X_raw['response_lag_min'].notna()]


X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)


X_train_num = X_train_raw[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
X_test_num  = X_test_raw[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")


# %%
# num_medians = X_train_num.median()                    # 你想改成均值/常數都可
# X_train_num = X_train_num.fillna(num_medians)
# X_test_num  = X_test_num.fillna(num_medians)

scaler = StandardScaler().fit(X_train_num)            # 想不用標準化，可把這段拿掉
X_train_num_scaled = pd.DataFrame(
    scaler.transform(X_train_num),
    columns=NUMERIC_COLS,
    index=X_train_num.index
)
X_test_num_scaled = pd.DataFrame(
    scaler.transform(X_test_num),
    columns=NUMERIC_COLS,
    index=X_test_num.index
)
# %%
X_train_cat = X_train_raw[CATEGORICAL_COLS].astype("string")
X_test_cat  = X_test_raw[CATEGORICAL_COLS].astype("string")

# %%
# 用 product_cate_missing 補缺
X_train_cat['Product_category'] = X_train_cat['Product_category'].fillna("missing")
X_test_cat['Product_category']  = X_test_cat['Product_category'].fillna("missing")

# %%
NEED_OHE = ["channel_name", "category", "Product_category", "Agent Shift"]
X_train_ohe = pd.get_dummies(X_train_cat[NEED_OHE], prefix=NEED_OHE, dummy_na=False)
X_test_ohe  = pd.get_dummies(X_test_cat[NEED_OHE],  prefix=NEED_OHE, dummy_na=False)

# 對齊欄位（測試集補齊缺欄、移除多餘欄）
X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

# %%
tenure_map_order = ["On Job Training", "0-30", "31-60", "61-90", ">90"]
# 建立順序對應表，例如：
tenure_map = {label: idx for idx, label in enumerate(tenure_map_order)}
print(tenure_map)

X_train_label = X_train_cat["Tenure Bucket"].map(tenure_map)
X_test_label = X_test_cat["Tenure Bucket"].map(tenure_map)



# X_test_pro = X_test_cat["Product_category"].map(lambda x: 1 if x != "missing" else 0)
# X_test_pro = X_test_cat["Product_category"].map(lambda x: 1 if x != "missing" else 0)

#%% 合併數值與類別
X_train = pd.concat([X_train_num_scaled, X_train_label, X_train_ohe], axis=1)
X_test  = pd.concat([X_test_num_scaled, X_test_label, X_test_ohe],  axis=1)


#%% 主模型：多類別 Logistic Regression（可解釋、係數方向性清楚）
feature_names = X_train.columns.tolist()

# 模型：多類別 Logistic Regression（可解釋）
logit = LogisticRegression(
    solver="saga",
    class_weight="balanced",
    max_iter=5000,
    n_jobs=-1,
)
logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)
print(f"[Logistic] acc={accuracy_score(y_test, y_pred):.3f}  macroF1={f1_score(y_test, y_pred, average='macro'):.3f}")
print(classification_report(y_test, y_pred, digits=3))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1], normalize="true")
cm

#%% 特徵重要性（Permutation Importance）
perm = permutation_importance(
    logit, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

imp_df = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# 80/20 原則
imp_df["importance_pos"] = imp_df["importance"].clip(lower=0)
total = imp_df["importance_pos"].sum()
imp_df["share"] = 0 if total == 0 else imp_df["importance_pos"] / total
imp_df["cum_share"] = imp_df["share"].cumsum()

top80 = imp_df.loc[imp_df["cum_share"] <= 0.8, "feature"].tolist()
if not top80:
    top80 = imp_df["feature"].head(12).tolist()

print("\n=== 覆蓋約 80% 影響度的關鍵特徵 ===")
for i, f in enumerate(top80, 1):
    print(f"{i:>2}. {f}")

# %%
# 準備「未標準化數值 + 同步 one-hot 類別」
# - X_train_num_raw / X_test_num_raw：未做標準化，只做缺失補中位數
X_train_num_raw = X_train_raw[NUMERIC_COLS]
X_test_num_raw  = X_test_raw[NUMERIC_COLS]

# 類別 one-hot（與先前相同，用訓練集欄位對齊）
X_train_rules = pd.concat([X_train_num_raw, X_train_label, X_train_ohe], axis=1)
X_test_rules  = pd.concat([X_test_num_raw, X_test_label, X_test_ohe],  axis=1)

# 用前面找出的 top80（或 imp_df 前幾名）
use_cols = [c for c in top80 if c in X_train_rules.columns]
# if len(use_cols) < 8:
#     extra = [c for c in imp_df["feature"].tolist() if c not in use_cols]
#     use_cols = (use_cols + extra)[:12]

dt_rules = DecisionTreeClassifier(
    max_depth=6, min_samples_leaf=60, class_weight="balanced", random_state=42
)
dt_rules.fit(X_train_rules[use_cols], y_train)
y_pred_dt = dt_rules.predict(X_test_rules[use_cols])

print(f"[Rules Tree - 原始單位] acc={accuracy_score(y_test, y_pred_dt):.3f}  f1={f1_score(y_test, y_pred_dt):.3f}")
print("\n=== 規則（max_depth=6" \
"；數值特徵已是『原始單位』） ===")
print(export_text(dt_rules, feature_names=list(use_cols)))

# %%
