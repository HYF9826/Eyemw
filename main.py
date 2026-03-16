import pandas as pd

# ================================
# 1 读取数据
# ================================
df = pd.read_excel("data/testdata_study1.xlsx")

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
    "AOIGazeProportion"
]

X_train = train_data[features]
y_train = train_data["TUTProbeResponse"]

X_test = test_data[features]
y_test = test_data["TUTProbeResponse"]


# ================================
# 4 Z-score 标准化
# 只用训练集 fit，避免数据泄漏
# ================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)   # 用训练集计算均值方差
X_test = scaler.transform(X_test)         # 用同样参数转换测试集

print("Z-score 标准化完成")


# ================================
# 5 Baseline 模型
# 最简单模型：始终预测多数类
# ================================
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score

print("Baseline Accuracy:", accuracy_score(y_test, baseline_pred))
print("Baseline F1:", f1_score(y_test, baseline_pred))  # 默认正类=1（走神）


# ================================
# 6 SMOTE 处理类别不平衡
# 只对训练集做过采样
# ================================
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy="auto",
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("SMOTE后类别分布：")
print(pd.Series(y_train_smote).value_counts())


# ================================
# 7 训练 SVM 模型
# ================================
from sklearn.svm import SVC

model = SVC(
    kernel="rbf",
    C=1.1,
    gamma=0.9,
    class_weight="balanced"  # 对不平衡数据更友好
)

model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)


# ================================
# 8 模型评估
# ================================
print("---------------------------------")

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)  # 正类=1（走神）

print("Accuracy:", acc)
print("F1:", f1)


# ================================
# 9 混淆矩阵
# ================================
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# ================================
# 10 AUC计算
# AUC需要使用decision score
# ================================
from sklearn.metrics import roc_auc_score

y_score = model.decision_function(X_test)

auc = roc_auc_score(y_test, y_score)

print("AUC:", auc)


# ================================
# 11 分类报告
# ================================
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))