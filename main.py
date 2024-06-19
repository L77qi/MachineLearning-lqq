# ----------导入需要的库----------
# 导入库
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb

# 参数搜索和评价的库
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ----------数据查看与分析----------
# 数据读取
Train_data = pd.read_csv('D:/0Knowledge/大二下/课程/机器学习/课设/used_car_train_20200313.csv', sep=' ')
TestB_data = pd.read_csv('D:/0Knowledge/大二下/课程/机器学习/课设/used_car_testB_20200421.csv', sep=' ')

# 查看数据的大小信息
print('Train data shape:', Train_data.shape)
print('TestB data shape:', TestB_data.shape)

# 简要浏览读取数据的形式
print('\n\nHead of the training dataset:')
print(Train_data.head())

# 查看对应一些数据列名，以及NAN缺失信息
print('\n\nInfo of the training dataset:')
print(Train_data.info())
print('\n\nInfo of the test dataset:')
print(TestB_data.info())

# 查看列名
print('\n\nColumns of the training dataset:')
print(Train_data.columns)

# 通过查看数值特征列的一些统计信息
print('\n\nStatistical description of the training dataset:')
print(Train_data.describe())
print('\n\nStatistical description of the test dataset:')
print(TestB_data.describe())


# ----------数据与特征构建----------
# 提取数值类型特征列名并输出
print('\n\nNumerical columns in the training dataset:')
numerical_cols = Train_data.select_dtypes(exclude='object').columns
print(numerical_cols)

# 提取类别类型特征列名并输出
print('\n\nCategorical columns in the training dataset:')
categorical_cols = Train_data.select_dtypes(include='object').columns
print(categorical_cols)

# 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID', 'name',
                                                             'creatData', 'price']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

# 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test = TestB_data[feature_cols]

print('\n\nX train shape:', X_data.shape)
print('X test shape:', X_test.shape)

# 定义一个统计函数
def Sta_inf(data):
    print('_min', np.min(data))
    print('_max', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))

print('\n\nSta of label:')
Sta_inf(Y_data)

# 绘制标签的统计图，查看标签分布
plt.hist(Y_data)
plt.title('Distribution of Labels', fontsize=16)
plt.show()
plt.close()

# 填补缺省值
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)


# ----------模型训练----------
# 选取xgb模型
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0,
                       subsample=0.8, colsample_bytree=0.9, max_depth=7)

scores_train = []  # 记录训练集上的MAE评分
scores = []  # 记录验证集上的MAE评分

# 选择5折交叉验证方式
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_ind, val_ind in sk.split(X_data, Y_data):

    train_x = X_data.iloc[train_ind].values
    train_y = Y_data.iloc[train_ind]
    val_x = X_data.iloc[val_ind].values
    val_y = Y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)  # 训练XGB模型
    pred_train_xgb = xgr.predict(train_x)  # 在训练集上进行预测
    pred_xgb = xgr.predict(val_x)  # 在验证集上进行预测

    score_train = mean_absolute_error(train_y, pred_train_xgb)  # 计算训练集上的MAE
    scores_train.append(score_train)
    score = mean_absolute_error(val_y, pred_xgb)  # 计算验证集上的MAE
    scores.append(score)

print('\n\nTrain mae:', np.mean(score_train))
print('Val mae:', np.mean(scores))

# 定义xgb和lgb模型函数
def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0,
                             subsample=0.8, colsample_bytree=0.9, max_depth=7)
    model.fit(x_train, y_train)  # 训练XGB模型
    return model

def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127, n_estimators=150, verbose=0)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid, verbose=0)
    gbm.fit(x_train, y_train)  # 使用网格搜索交叉验证寻找最佳参数
    return gbm

# 切分数据集进行模型训练，评价和预测
x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.3)

print('\n\nTrain lgb...')
model_lgb = build_model_lgb(x_train, y_train)  # 训练LGB模型
val_lgb = model_lgb.predict(x_val)  # 在验证集上进行预测
MAE_lgb = mean_absolute_error(y_val, val_lgb)
print('MAE of val with lgb:', MAE_lgb)

print('\n\nPredict lgb...')
model_lgb_pre = build_model_lgb(X_data, Y_data)  # 使用整个数据集训练LGB模型
subA_lgb = model_lgb_pre.predict(X_test)  # 对测试集进行预测
print('Sta of Predict lgb:')
Sta_inf(subA_lgb)

print('\n\nTrain xgb...')
model_xgb = build_model_xgb(x_train, y_train)  # 训练XGB模型
val_xgb = model_xgb.predict(x_val)  # 在验证集上进行预测
MAE_xgb = mean_absolute_error(y_val, val_xgb)
print('MAE of val with xgb:', MAE_xgb)

print('\n\nPredict xgb...')
model_xgb_pre = build_model_xgb(X_data, Y_data)  # 使用整个数据集训练XGB模型
subA_xgb = model_xgb_pre.predict(X_test)  # 对测试集进行预测
print('Sta of Predict xgb:')
Sta_inf(subA_xgb)

# 进行模型的加权融合
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb
val_Weighted[val_Weighted<0] = 10  # 处理加权融合后的结果
print('\n\nMAE of val with Weighted ensemble:', mean_absolute_error(y_val, val_Weighted))

sub_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*subA_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*subA_xgb

# 查看预测值的统计进行
plt.hist(Y_data)
plt.title('Distribution of Predicted Values', fontsize=16)
plt.show()
plt.close()

# 得出结果
sub = pd.DataFrame()
sub['SaleID'] = TestB_data.SaleID
sub['price'] = sub_Weighted
sub.to_csv('./sub_Weighted.csv', index=False)

print(sub.head())
