import warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

# ================================
# 数据路径（可按需改为 data/testdata.xlsx 等）
# ================================
DATA_PATH = "data/testdata.xlsx"

# 七个眼动特征列（与 Excel 表头一致）
FEATURE_COLS = [
    "Gazes",
    "UniqueGazes",
    "UniqueGazeProportion",
    "OffscreenGazes",
    "OffScreenGazeProportion",
    "AOIGazes",
    "AOIGazeProportion",
]


class TrainCentroid1DistanceAugment(BaseEstimator, TransformerMixin):
    """
    仅在当前 fit 所见的训练数据上：
    1) StandardScaler 拟合于 X（7 维）
    2) 在标准化空间用标签 y==1 的样本求「全局 1 类中心」mu_1（均值向量）
    3) transform：输出 [Z | ||Z - mu_1||_2]，共 8 维
    测试集仅 transform，使用训练时拟合的 scaler 与 mu_1，避免数据泄漏。
    """

    def __init__(self):
        self.scaler_ = None
        self.mu1_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TrainCentroid1DistanceAugment 需要标签 y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X)
        m1 = y == 1
        if m1.sum() == 0:
            self.mu1_ = Z.mean(axis=0)
        else:
            self.mu1_ = Z[m1].mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = self.scaler_.transform(X)
        d = np.linalg.norm(Z - self.mu1_, axis=1, keepdims=True)
        return np.hstack([Z, d])


# ================================
# 1 读取数据
# ================================
df = pd.read_excel(DATA_PATH)

for c in FEATURE_COLS + ["TUTProbeResponse", "ParticipantNum"]:
    if c not in df.columns:
        raise SystemExit(f"缺少列: {c}")

X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
y = df["TUTProbeResponse"].astype(int)
groups = df["ParticipantNum"]

print("数据:", DATA_PATH, "形状:", X.shape)
print("特征: 7 维眼动 + 1 维「到训练集 1 类中心」欧氏距离（在每折内计算）")

# ================================
# 2 外层：按被试 10 折
# ================================
outer_cv = GroupKFold(n_splits=10)

# ================================
# 3 管线：填补 -> 标准化+1类中心距离 -> SMOTE -> SVM
# ================================
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("centroid1", TrainCentroid1DistanceAugment()),
        ("smote", SMOTE(random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
    ]
)

param_grid = {
    "smote__k_neighbors": [5],
    "svm__kernel": ["rbf"],
    "svm__C": [1, 10],
    "svm__gamma": ["scale", 0.1],
    "svm__class_weight": ["balanced"],
}

inner_cv = GroupKFold(n_splits=5)

fold_kappa = []
fold_auc = []
per_class_precision_0 = []
per_class_recall_0 = []
per_class_f1_0 = []
per_class_precision_1 = []
per_class_recall_1 = []
per_class_f1_1 = []

for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    fold_kappa.append(kappa)
    fold_auc.append(auc)
    per_class_precision_0.append(report["0"]["precision"])
    per_class_recall_0.append(report["0"]["recall"])
    per_class_f1_0.append(report["0"]["f1-score"])
    per_class_precision_1.append(report["1"]["precision"])
    per_class_recall_1.append(report["1"]["recall"])
    per_class_f1_1.append(report["1"]["f1-score"])

    print(f"\n================ Fold {fold_id} ================")
    print("Best Params:", search.best_params_)
    print("Best Inner-CV F1:", round(search.best_score_, 4))
    print(
        f"Class 0 -> P:{report['0']['precision']:.4f} R:{report['0']['recall']:.4f} F1:{report['0']['f1-score']:.4f}"
    )
    print(
        f"Class 1 -> P:{report['1']['precision']:.4f} R:{report['1']['recall']:.4f} F1:{report['1']['f1-score']:.4f}"
    )
    print("Kappa:", round(kappa, 4))
    print("AUC:", round(auc, 4))


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


p0_m, p0_s = mean_std(per_class_precision_0)
r0_m, r0_s = mean_std(per_class_recall_0)
f0_m, f0_s = mean_std(per_class_f1_0)
p1_m, p1_s = mean_std(per_class_precision_1)
r1_m, r1_s = mean_std(per_class_recall_1)
f1_m, f1_s = mean_std(per_class_f1_1)
k_m, k_s = mean_std(fold_kappa)
auc_m, auc_s = mean_std(fold_auc)

print("\n===============================================")
print("10-Fold GroupCV Summary (SVM, 7 gaze + dist to train global class-1 center)")
print("Class 0 Precision: {:.4f} ± {:.4f}".format(p0_m, p0_s))
print("Class 0 Recall:    {:.4f} ± {:.4f}".format(r0_m, r0_s))
print("Class 0 F1:        {:.4f} ± {:.4f}".format(f0_m, f0_s))
print("Class 1 Precision: {:.4f} ± {:.4f}".format(p1_m, p1_s))
print("Class 1 Recall:    {:.4f} ± {:.4f}".format(r1_m, r1_s))
print("Class 1 F1:        {:.4f} ± {:.4f}".format(f1_m, f1_s))
print("Kappa:             {:.4f} ± {:.4f}".format(k_m, k_s))
print("AUC:               {:.4f} ± {:.4f}".format(auc_m, auc_s))
