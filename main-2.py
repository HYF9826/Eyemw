import pandas as pd

# ================================
# 1 读取数据
# ================================
df = pd.read_excel("data/testdata_study2.xlsx")

# 标签说明
# 1 = mind wandering (正类)
# 0 = on-task


# ================================
# 2 按被试划分训练集和测试集
# 避免同一被试同时出现在train/test
# ================================
subjects = df["ParticipantNum"].unique()

from sklearn.model_selection import train_test_split

train_subjects, test_subjects = train_test_split(
    subjects,
    test_size=0.3,
    random_state=42
)

train_data = df[df["ParticipantNum"].isin(train_subjects)]
test_data = df[df["ParticipantNum"].isin(test_subjects)]

# ================================
# 3 选择特征和标签
# ================================
features = [
    "Gazes",
    "UniqueGazes",
    "UniqueGazeProportion",
    "OffscreenGazes",
    "OffScreenGazeProportion",
    "AOIGazes",
    "AOIGazeProportion",
    "GazeProficiency"
]

X_train = train_data[features]
y_train = train_data["TUTProbeResponse"]

X_test = test_data[features]
y_test = test_data["TUTProbeResponse"]

# ================================
# 4 Baseline 模型
# 最简单模型：始终预测多数类
# ================================
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("Baseline Accuracy:", accuracy_score(y_test, baseline_pred))
print("Baseline F1:", f1_score(y_test, baseline_pred))

# ================================
# 5 构建机器学习 Pipeline
# Pipeline 可以避免数据泄漏
# 每一折交叉验证都会重新执行：
# SMOTE → 标准化 → SVM
# ================================

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([

    # Step1：SMOTE过采样（只对训练数据）
    ("smote", SMOTE(random_state=42)),

    # Step2：Z-score标准化
    ("scaler", StandardScaler()),

    # Step3：SVM分类器
    ("svm", SVC(
        kernel="rbf",
        C=1.1,
        gamma=0.9,
        probability=True  # 为了后面计算AUC
    ))
])

# ================================
# 6 10折交叉验证（GroupKFold）
# 按被试进行交叉验证
# 防止同一被试数据同时出现在训练和验证集中
# ================================

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score

groups = train_data["ParticipantNum"]

gkf = GroupKFold(n_splits=10)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=gkf,
    groups=groups,
    scoring="f1"
)

print("10-fold Group CV F1:", cv_scores.mean())

# ================================
# 7 在整个训练集上训练最终模型
# ================================

pipeline.fit(X_train, y_train)

# ================================
# 8 在测试集上预测
# ================================

y_pred = pipeline.predict(X_test)

# 预测概率（用于AUC）
y_prob = pipeline.predict_proba(X_test)[:, 1]

# ================================
# 9 模型评估
# ================================

print("---------------------------------")

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", acc)
print("F1:", f1)

# ================================
# 10 混淆矩阵
# ================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# ================================
# 11 AUC
# ================================

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_prob)

print("AUC:", auc)

# ================================
# 12 分类报告
# ================================

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))