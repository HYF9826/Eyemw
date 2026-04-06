import warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
# ================================
# 1 读取数据并定义特征
# ================================
# 目标：基于眼动特征识别走神（1=mind wandering，0=on-task）
df = pd.read_excel("data/testdata_study3.xlsx")

features = [
    "Gazes",
    "UniqueGazes",
    "UniqueGazeProportion",
    "OffscreenGazes",
    "OffScreenGazeProportion",
    "AOIGazes",
    "AOIGazeProportion",
    "UniqueSpeed1",
    "OffscreenSpeed1",
    "AOISpeed1"

]

X = df[features]
y = df["TUTProbeResponse"].astype(int)
groups = df["ParticipantNum"]

# ================================
# 2 定义外层10折分组交叉验证（按被试）
# ================================
# 外层CV用于最终泛化评估，保证同一被试不会同时出现在训练和测试中
outer_cv = GroupKFold(n_splits=10)

# ================================
# 3 定义内层调参流程（标准化 -> SMOTE -> SVM）
# ================================
# 每次拟合时：
# - 标准化参数仅由训练折计算并应用到验证/测试折
# - SMOTE仅在训练折执行，验证/测试保持原始分布
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("svm", SVC(probability=True, random_state=42)),
])

# SVM参数网格（可按计算资源增减）
param_grid = {
    # 缩小网格以减少搜索次数和总训练时长
    "smote__k_neighbors": [5],
    "svm__kernel": ["rbf"],
    "svm__C": [1, 10],
    "svm__gamma": ["scale", 0.1],
    "svm__class_weight": ["balanced"],
}

# 内层分组CV仅用于超参数搜索
inner_cv = GroupKFold(n_splits=5)

# ================================
# 4 外层评估循环：每折内做网格搜索并测试
# ================================
fold_kappa = []
fold_auc = []

# 每类指标（按测试折汇总）
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

    # 在当前外层训练折内进行分组网格搜索（避免泄漏）
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

    # 使用当前外层测试折评估
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
    print(f"Class 0 -> P:{report['0']['precision']:.4f} R:{report['0']['recall']:.4f} F1:{report['0']['f1-score']:.4f}")
    print(f"Class 1 -> P:{report['1']['precision']:.4f} R:{report['1']['recall']:.4f} F1:{report['1']['f1-score']:.4f}")
    print("Kappa:", round(kappa, 4))
    print("AUC:", round(auc, 4))

# ================================
# 5 汇总并输出最终结果（均值±标准差）
# ================================
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
print("10-Fold GroupCV Summary (SVM only)")
print("Class 0 Precision: {:.4f} ± {:.4f}".format(p0_m, p0_s))
print("Class 0 Recall:    {:.4f} ± {:.4f}".format(r0_m, r0_s))
print("Class 0 F1:        {:.4f} ± {:.4f}".format(f0_m, f0_s))

print("Class 1 Precision: {:.4f} ± {:.4f}".format(p1_m, p1_s))
print("Class 1 Recall:    {:.4f} ± {:.4f}".format(r1_m, r1_s))
print("Class 1 F1:        {:.4f} ± {:.4f}".format(f1_m, f1_s))

print("Kappa:             {:.4f} ± {:.4f}".format(k_m, k_s))
print("AUC:               {:.4f} ± {:.4f}".format(auc_m, auc_s))