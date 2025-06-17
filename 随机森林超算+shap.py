import sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy import stats as spstats
from tqdm import tqdm
import time
import pickle
import warnings
import json
import argparse  # 添加到导入部分
# 在现有导入部分添加
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from shapely import wkt
import geopandas as gpd

import traceback
import glob
warnings.filterwarnings('ignore')

import matplotlib.font_manager as fm



import matplotlib.font_manager as fm


font_path = "/share/home/tdang25/noto_fonts/NotoSansCJKsc-Regular.otf"

# 注册字体
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

# 设置 matplotlib 默认字体
plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print(f"✅ 成功设置 matplotlib 中文字体为: {font_name}")



# 全局变量定义
# 特征组1：气候变量
FEATURE_COLUMNS1 = ['rx1day','r10mm','cdd','tx10pd','r99totd','sdiid','cddd','rx5d','tn90p','tx90pd','rx1dayd','r99tot',
'r95tot','tn10pd','rx5dd','tn90pd','txnd','txn','tn10p','sdii','tx10p','dtrd','cwd','txxd','tnn','r95totd','r10mmd','r20mm','tnnd',
'tnx','id_1d','cwdd']

# 特征组2：全部变量
FEATURE_COLUMNS2 = ['rk', 'bts', 'ld', 'nt','dem','water','slope','hli', 'rkd', 'btsd', 'ldd', 'ntd', 'waterd',
'bio6d','rx1day','r10mm','bio19','bio8','bio6','cdd','bio13d',
'bio17d','bio17','bio3d','bio19d','tx10pd','r99totd','sdiid','bio15','cddd','rx5d',
'bio7','tn90p','tx90pd','rx1dayd','r99tot','bio14d','bio9d','r95tot',
'tn10pd','bio14','rx5dd','tn90pd','bio4','bio3','txnd','bio15d','bio13',
'tx90p','r20mmd','txn','tn10p','sdii','tx10p','bio4d','dtrd','bio9','bio2d',
'cwd','txxd','tnn','bio18','bio7d','r95totd','r10mmd','r20mm','bio8d','bio18d',
'tnnd','tnx','id_1d','cwdd']

# 特征组3：其他变量
FEATURE_COLUMNS3 = ['rk', 'bts', 'ld', 'nt','dem','water','slope','hli', 'rkd', 'btsd', 'ldd', 'ntd', 'waterd',
'bio6d','waterd','slope','bio19','bio8','bio6','bio13d','bio17d','bio17','bio3d','bio19d','bio15',
'bio7','bio14d','bio9d','ld','bio14','ldd','bio4','bio3','bio15d','bio13','bio4d','bio9','bio2d','bio18','bio7d','bio8d','bio18d'
]

# 特征组4：筛选变量（原全部个变量）
FEATURE_COLUMNS4 = [ 'rk', 'bts', 'ld', 'nt', 'dem', 'slope',  'hli', 'water'
                      
                      , 'cdd', 'cwd', 'dtr', 'fd', 'id_1', 'prcptot', 'r10mm', 'r20mm', 'r95tot'
                      , 'r99tot', 'rx1day', 'rx5d', 'sdii', 'su', 'tn10p', 'tn90p', 'tnn', 'tnx', 'tr', 'tx10p', 'tx90p', 'txn', 'txx', 
                      
                      'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai7', 'ai8', 'ca1', 'ca2', 'ca3', 'ca4', 'ca5', 'ca7', 'ca8',
                      'con1', 'con2', 'con3', 'con4', 'con5', 'con7', 'con8', 'cot1', 'cot2', 'cot3', 'cot4', 'cot5', 'cot7', 'cot8', 'cog',
                      'din1', 'din2', 'din3', 'din4', 'din5', 'din7', 'din8',
                      'iji1', 'iji2', 'iji3', 'iji4', 'iji5', 'iji7', 'iji8',
                      'lpi1', 'lpi2', 'lpi3', 'lpi4', 'lpi5', 'lpi7', 'lpi8',
                      'lsi1', 'lsi2', 'lsi3', 'lsi4', 'lsi5', 'lsi7', 'lsi8',
                      'np1', 'np2', 'np3', 'np4', 'np5', 'np7', 'np8',
                      'pd1', 'pd2', 'pd3', 'pd4', 'pd5', 'pd7', 'pd8',
                      'pladj1', 'pladj2', 'pladj3', 'pladj4', 'pladj5', 'pladj7', 'pladj8',
                      'pland1', 'pland2', 'pland3', 'pland4', 'pland5', 'pland7', 'pland8',
                      'pr', 'shdi', 'sidi', 'split1', 'split2', 'split3', 'split4', 'split5', 'split7', 'split8', 'ta',
                      'tca1', 'tca2', 'tca3', 'tca4', 'tca5', 'tca7', 'tca8',
                      'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19',
                      
                        'rkd', 'btsd', 'ldd', 'ntd', 'waterd'
                      
                      , 'cddd', 'cwdd', 'dtrd', 'fdd', 'id_1d', 'prcptotd', 'r10mmd', 'r20mmd', 'r95totd', 'r99totd'
                      , 'rx1dayd', 'rx5dd', 'sdiid', 'sud', 'tn10pd', 'tn90pd', 'tnnd', 'tnxd', 'trd', 'tx10pd', 'tx90pd', 'txnd', 'txxd', 
                      
                      'ai1d', 'ai2d', 'ai3d', 'ai4d', 'ai5d', 'ai7d', 'ai8d', 'ca1d', 'ca2d', 'ca3d', 'ca4d', 'ca5d', 'ca7d', 'ca8d',
                      'con1d', 'con2d', 'con3d', 'con4d', 'con5d', 'con7d', 'con8d', 'cot1d', 'cot2d', 'cot3d', 'cot4d', 'cot5d', 'cot7d', 'cot8d', 'cogd',
                      'din1d', 'din2d', 'din3d', 'din4d', 'din5d', 'din7d', 'din8d',
                      'iji1d', 'iji2d', 'iji3d', 'iji4d', 'iji5d', 'iji7d', 'iji8d',
                      'lpi1d', 'lpi2d', 'lpi3d', 'lpi4d', 'lpi5d', 'lpi7d', 'lpi8d',
                      'lsi1d', 'lsi2d', 'lsi3d', 'lsi4d', 'lsi5d', 'lsi7d', 'lsi8d',
                      'np1d', 'np2d', 'np3d', 'np4d', 'np5d', 'np7d', 'np8d',
                      'pd1d', 'pd2d', 'pd3d', 'pd4d', 'pd5d', 'pd7d', 'pd8d',
                      'pladj1d', 'pladj2d', 'pladj3d', 'pladj4d', 'pladj5d', 'pladj7d', 'pladj8d',
                      'pland1d', 'pland2d', 'pland3d', 'pland4d', 'pland5d', 'pland7d', 'pland8d',
                      'prd', 'shdid', 'sidid', 'split1d', 'split2d', 'split3d', 'split4d', 'split5d', 'split7d', 'split8d', 'tad',
                      'tca1d', 'tca2d', 'tca3d', 'tca4d', 'tca5d', 'tca7d', 'tca8d',
                      'bio1d', 'bio2d', 'bio3d', 'bio4d', 'bio5d', 'bio6d', 'bio7d', 'bio8d'
                      , 'bio9d', 'bio10d', 'bio11d', 'bio12d', 'bio13d', 'bio14d', 'bio15d', 'bio16d', 'bio17d', 'bio18d', 'bio19d'
]

# 定义特征组名称映射
FEATURE_GROUPS = {
    'climate_vars': {'name': '气候变量', 'features': FEATURE_COLUMNS1},
    'all_vars': {'name': '全部变量', 'features': FEATURE_COLUMNS2},
    'other_vars': {'name': '其他变量', 'features': FEATURE_COLUMNS3},
    'selected_vars': {'name': '筛选变量', 'features': FEATURE_COLUMNS4}
}

# 定义需要排除的列
EXCLUDE_COLUMNS = ['ID', 'geometry', 'x', 'y', 'orient','daolu','daolud','wsdi','wsdid','dg','dgd']

# 文件相关全局变量
DATA_DIR = r"/share/home/tdang25/6.12qihou/csv_data/all305_none_clean"
OUTPUT_DIR = r"/share/home/tdang25/6.12qihou/rf_model_4"

# SHAP分析配置
N_CLUSTERS = 72342  # 用于K-means聚类的背景样本数量
BATCH_SIZE_SHAP = 500  # SHAP计算的批量大小
N_JOBS = 18  # 并行计算的工作线程数

CLIMATE_PAIRS = [
    ('cdd', 'cddd'), ('cwd', 'cwdd'), ('dtr', 'dtrd'), ('fd', 'fdd'),
    ('id_1', 'id_1d'), ('prcptot', 'prcptotd'), ('r10mm', 'r10mmd'), ('r20mm', 'r20mmd'),
    ('r95tot', 'r95totd'), ('r99tot', 'r99totd'), ('rx1day', 'rx1dayd'), ('rx5d', 'rx5dd'),
    ('sdii', 'sdiid'), ('su', 'sud'), ('tn10p', 'tn10pd'), ('tn90p', 'tn90pd'),
    ('tnn', 'tnnd'), ('tnx', 'tnxd'), ('tr', 'trd'), ('tx10p', 'tx10pd'), ('tx90p', 'tx90pd'),
    ('txn', 'txnd'), ('txx', 'txxd')    ]
#全部变量2——对子变量
ALL_PAIRS = [ ('rk', 'rkd'), ('bts', 'btsd'), ('ld', 'ldd'), ('nt', 'ntd'),
    ('water', 'waterd'), ('cdd', 'cddd'), ('cwd', 'cwdd'), ('dtr', 'dtrd'), ('fd', 'fdd'),
    ('id_1', 'id_1d'), ('prcptot', 'prcptotd'), ('r10mm', 'r10mmd'), ('r20mm', 'r20mmd'),
    ('r95tot', 'r95totd'), ('r99tot', 'r99totd'), ('rx1day', 'rx1dayd'), ('rx5d', 'rx5dd'),
    ('sdii', 'sdiid'), ('su', 'sud'), ('tn10p', 'tn10pd'), ('tn90p', 'tn90pd'),
    ('tnn', 'tnnd'), ('tnx', 'tnxd'), ('tr', 'trd'), ('tx10p', 'tx10pd'), ('tx90p', 'tx90pd'),
    ('txn', 'txnd'), ('txx', 'txxd'), ('ai1', 'ai1d'), ('ai2', 'ai2d'), ('ai3', 'ai3d'),
    ('ai4', 'ai4d'), ('ai5', 'ai5d'), ('ai7', 'ai7d'), ('ai8', 'ai8d'), ('ca1', 'ca1d'),
    ('ca2', 'ca2d'), ('ca3', 'ca3d'), ('ca4', 'ca4d'), ('ca5', 'ca5d'), ('ca7', 'ca7d'),
    ('ca8', 'ca8d'), ('con1', 'con1d'), ('con2', 'con2d'), ('con3', 'con3d'), ('con4', 'con4d'),
    ('con5', 'con5d'), ('con7', 'con7d'), ('con8', 'con8d'), ('cot1', 'cot1d'), ('cot2', 'cot2d'),
    ('cot3', 'cot3d'), ('cot4', 'cot4d'), ('cot5', 'cot5d'), ('cot7', 'cot7d'), ('cot8', 'cot8d'),
    ('cog', 'cogd'), ('din1', 'din1d'), ('din2', 'din2d'), ('din3', 'din3d'), ('din4', 'din4d'),
    ('din5', 'din5d'), ('din7', 'din7d'), ('din8', 'din8d'), ('iji1', 'iji1d'), ('iji2', 'iji2d'),
    ('iji3', 'iji3d'), ('iji4', 'iji4d'), ('iji5', 'iji5d'), ('iji7', 'iji7d'), ('iji8', 'iji8d'),
    ('lpi1', 'lpi1d'), ('lpi2', 'lpi2d'), ('lpi3', 'lpi3d'), ('lpi4', 'lpi4d'), ('lpi5', 'lpi5d'),
    ('lpi7', 'lpi7d'), ('lpi8', 'lpi8d'), ('lsi1', 'lsi1d'), ('lsi2', 'lsi2d'), ('lsi3', 'lsi3d'),
    ('lsi4', 'lsi4d'), ('lsi5', 'lsi5d'), ('lsi7', 'lsi7d'), ('lsi8', 'lsi8d'), ('np1', 'np1d'),
    ('np2', 'np2d'), ('np3', 'np3d'), ('np4', 'np4d'), ('np5', 'np5d'), ('np7', 'np7d'),
    ('np8', 'np8d'), ('pd1', 'pd1d'), ('pd2', 'pd2d'), ('pd3', 'pd3d'), ('pd4', 'pd4d'),
    ('pd5', 'pd5d'), ('pd7', 'pd7d'), ('pd8', 'pd8d'), ('pladj1', 'pladj1d'), ('pladj2', 'pladj2d'),
    ('pladj3', 'pladj3d'), ('pladj4', 'pladj4d'), ('pladj5', 'pladj5d'), ('pladj7', 'pladj7d'),
    ('pladj8', 'pladj8d'), ('pland1', 'pland1d'), ('pland2', 'pland2d'), ('pland3', 'pland3d'),
    ('pland4', 'pland4d'), ('pland5', 'pland5d'), ('pland7', 'pland7d'), ('pland8', 'pland8d'),
    ('pr', 'prd'), ('shdi', 'shdid'), ('sidi', 'sidid'), ('split1', 'split1d'), ('split2', 'split2d'),
    ('split3', 'split3d'), ('split4', 'split4d'), ('split5', 'split5d'), ('split7', 'split7d'),
    ('split8', 'split8d'), ('ta', 'tad'), ('tca1', 'tca1d'), ('tca2', 'tca2d'), ('tca3', 'tca3d'),
    ('tca4', 'tca4d'), ('tca5', 'tca5d'), ('tca7', 'tca7d'), ('tca8', 'tca8d'), ('bio1', 'bio1d'),
    ('bio2', 'bio2d'), ('bio3', 'bio3d'), ('bio4', 'bio4d'), ('bio5', 'bio5d'), ('bio6', 'bio6d'),
    ('bio7', 'bio7d'), ('bio8', 'bio8d'), ('bio9', 'bio9d'), ('bio10', 'bio10d'), ('bio11', 'bio11d'),
    ('bio12', 'bio12d'), ('bio13', 'bio13d'), ('bio14', 'bio14d'), ('bio15', 'bio15d'),
    ('bio16', 'bio16d'), ('bio17', 'bio17d'), ('bio18', 'bio18d'), ('bio19', 'bio19d')
]
# 非对子变量（独立变量）
NON_PAIR_FEATURES = ['dem', 'slope', 'hli']

FILE_LIST = [
    "热点01_00.csv", "热点02_01.csv", "热点03_02.csv", "热点04_03.csv", 
    "热点05_04.csv", "热点06_05.csv", "热点07_06.csv", "热点08_07.csv", 
    "热点09_08.csv", "热点10_09.csv", "热点11_10.csv", "热点13_11.csv", 
    "热点14_13.csv", "热点15_14.csv", "热点16_15.csv", "热点17_16.csv", 
    "热点18_17.csv", "热点19_18.csv"
]

# 创建目录结构
def create_directory_structure():
    """创建结果存储的目录结构"""
    # 主目录
    dirs = [OUTPUT_DIR, 
           os.path.join(OUTPUT_DIR, 'climate_vars'), 
           os.path.join(OUTPUT_DIR, 'all_vars'), 
           os.path.join(OUTPUT_DIR, 'other_vars'), 
           os.path.join(OUTPUT_DIR, 'selected_vars'),
           os.path.join(OUTPUT_DIR, 'model_comparison')]
    
    # 为每个模型创建子目录
    for model_dir in dirs[1:5]:
        subdirs = [
            os.path.join(model_dir, 'plots'),
            os.path.join(model_dir, 'metrics'),
            os.path.join(model_dir, 'importance'),
            os.path.join(model_dir, 'models')  # 新增模型保存目录
        ]
        dirs.extend(subdirs)
    
    # 创建所有目录
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print(f"已创建结果存储目录结构在 {OUTPUT_DIR}")
    return dirs

# 加载数据
# 修改现有的load_data函数
def load_data(file_paths):
    """加载并合并多个CSV文件，并添加年份信息"""
    print("正在加载多个CSV文件的数据...")
    all_data = []
    file_years = []  # 用于记录文件名和年份映射
    
    # 使用进度条显示加载过程
    for i, file in enumerate(tqdm(file_paths, desc="加载CSV文件")):
        try:
            df = pd.read_csv(file)
            
            # 添加年份标识
            # === 添加年份标识 ===
            if i < 11:
                year = 2001 + i  # 2001到2011
            elif i == 11:
                year = 2013  # 跳过2012
            else:
                year = 2001 + i + 1  # 2013及之后
                
            df['year'] = year
            df['ID_year'] = df['ID'].astype(str) + "_" + str(year)
            
            file_name = os.path.basename(file)
            file_years.append((file_name, year))
            
            all_data.append(df)
            print(f"已加载 {file}：{df.shape[0]} 行 {df.shape[1]} 列 (年份: {year})")
        except Exception as e:
            print(f"加载 {file} 出错: {e}")
    
    # 打印文件与年份映射
    print("\n===== 文件与年份映射 =====")
    for file_name, year in file_years:
        print(f"{file_name} → {year}年")
    
    if all_data:
        merged_data = pd.concat(all_data, ignore_index=True)
        print(f"总数据：{merged_data.shape[0]} 行，{merged_data.shape[1]} 列")
        return merged_data
    else:
        print("未加载任何数据！")
        return None

# 准备特征数据
def prepare_features(data, feature_list, target_col='agbd', exclude_columns=None):
    """准备用于建模的特征和目标变量"""
    if exclude_columns is None:
        exclude_columns = []
    
    # 过滤有效特征（确保在数据集中存在且不在排除列表中）
    valid_features = [f for f in feature_list if f in data.columns and f not in exclude_columns]
    missing_features = [f for f in feature_list if f not in data.columns]
    
    if missing_features:
        print(f"警告：以下特征在数据集中不存在：{missing_features}")
    
    # 创建特征(X)和目标变量(y)
    X = data[valid_features]
    y = data[target_col]
    
    return X, y, valid_features

# 计算调整后的R²
def calculate_adjusted_r2(r2, n, p):
    """计算调整后的R²
    
    参数:
        r2: R²值
        n: 样本数
        p: 特征数
        
    返回:
        调整后的R²值
    """
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# 训练和评估随机森林模型
def train_evaluate_rf_model(X, y, n_folds=5, test_size=0.2, random_state=27):
    """训练随机森林模型并使用交叉验证进行评估"""
    print("开始训练和评估随机森林模型...")
    
    # 分割训练集和测试集（用于验证模型）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 初始化K折交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # 存储评估指标的列表
    fold_metrics = []
    all_fold_y_pred = []
    all_fold_y_true = []
    
    # 存储每个折叠的实际和预测值，用于t检验
    fold_actual_pred = {}
    
    # 存储每个折的模型
    fold_models = {}
    
    # 执行交叉验证
    print(f"正在执行{n_folds}折交叉验证...")
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train_scaled), total=n_folds, desc="交叉验证进度"), 1):
        # 获取当前折叠的训练和验证数据
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 训练模型
        rf_model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
        rf_model.fit(X_fold_train, y_fold_train)
        
        # 保存该折的模型
        fold_models[f'fold_{fold}'] = rf_model
        
        # 在验证集上预测
        y_fold_pred = rf_model.predict(X_fold_val)
        
        # 存储数据用于后续分析
        all_fold_y_pred.extend(y_fold_pred)
        all_fold_y_true.extend(y_fold_val.values)
        fold_actual_pred[f'fold_{fold}'] = {'actual': y_fold_val.values, 'predicted': y_fold_pred}
        
        # 计算评估指标
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        fold_adj_r2 = calculate_adjusted_r2(fold_r2, len(y_fold_val), X.shape[1])
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
        fold_mse = mean_squared_error(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_bias = np.mean(y_fold_pred - y_fold_val)
        
        # 计算当前折叠的特征重要性
        fold_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # 存储评估指标
        fold_metrics.append({
            'fold': fold,
            'R²': fold_r2,
            'Adjusted_R²': fold_adj_r2,
            'RMSE': fold_rmse,
            'MSE': fold_mse,
            'MAE': fold_mae,
            'Bias': fold_bias,
            'Feature_Importances': fold_importances
        })
    
    # 计算交叉验证平均指标
    mean_cv_r2 = np.mean([metrics['R²'] for metrics in fold_metrics])
    mean_cv_adj_r2 = np.mean([metrics['Adjusted_R²'] for metrics in fold_metrics])
    mean_cv_rmse = np.mean([metrics['RMSE'] for metrics in fold_metrics])
    mean_cv_mse = np.mean([metrics['MSE'] for metrics in fold_metrics])
    mean_cv_mae = np.mean([metrics['MAE'] for metrics in fold_metrics])
    mean_cv_bias = np.mean([metrics['Bias'] for metrics in fold_metrics])
    
    print("正在训练验证模型（使用全部训练数据）...")
    # 使用全部训练数据训练验证模型
    validation_model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    validation_model.fit(X_train_scaled, y_train)
    
    # 在测试集（验证数据）上预测
    val_y_pred = validation_model.predict(X_test_scaled)
    
    # 计算验证模型的评估指标
    val_r2 = r2_score(y_test, val_y_pred)
    val_adj_r2 = calculate_adjusted_r2(val_r2, len(y_test), X.shape[1])
    val_rmse = np.sqrt(mean_squared_error(y_test, val_y_pred))
    val_mse = mean_squared_error(y_test, val_y_pred)
    val_mae = mean_absolute_error(y_test, val_y_pred)
    val_bias = np.mean(val_y_pred - y_test)
    
    # 计算验证模型的特征重要性
    val_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': validation_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("正在训练全数据模型（使用全部数据）...")
    # 使用全部数据训练模型
    all_data_model = RandomForestRegressor(n_estimators=500, random_state=27, n_jobs=-1)

    # 标准化全部数据
    X_all_scaled = scaler.fit_transform(X)

    # 不再分割数据，而是使用全部数据进行训练
    all_data_model.fit(X_all_scaled, y)

    # 在相同的数据上进行评估（内部评估）
    all_data_y_pred = all_data_model.predict(X_all_scaled)

    # 计算全数据模型的评估指标
    all_data_r2 = r2_score(y, all_data_y_pred)
    all_data_adj_r2 = calculate_adjusted_r2(all_data_r2, len(y), X.shape[1])
    all_data_rmse = np.sqrt(mean_squared_error(y, all_data_y_pred))
    all_data_mse = mean_squared_error(y, all_data_y_pred)
    all_data_mae = mean_absolute_error(y, all_data_y_pred)
    all_data_bias = np.mean(all_data_y_pred - y)
    
    # 计算全数据模型的特征重要性
    all_data_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': all_data_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 存储每个折叠的实际值和预测值，用于t检验
    fold_actual_pred['validation'] = {'actual': y_test.values, 'predicted': val_y_pred}
    fold_actual_pred['all_data'] = {'actual': y.values, 'predicted': all_data_y_pred}
    
    # 进行R²和RMSE的配对t检验
    r2_values = [metrics['R²'] for metrics in fold_metrics]
    adj_r2_values = [metrics['Adjusted_R²'] for metrics in fold_metrics]
    rmse_values = [metrics['RMSE'] for metrics in fold_metrics]
    
    # R²与调整后R²的t检验
    r2_vs_adj_r2_ttest = stats.ttest_rel(r2_values, adj_r2_values)
    
    print("模型训练和评估完成")
    
    # 创建完整结果字典
    results = {
        'fold_metrics': pd.DataFrame(fold_metrics),
        'fold_models': fold_models,  # 现在包含了每个折叠的模型
        'mean_cv_metrics': {
            'R²': mean_cv_r2,
            'Adjusted_R²': mean_cv_adj_r2,
            'RMSE': mean_cv_rmse,
            'MSE': mean_cv_mse,
            'MAE': mean_cv_mae,
            'Bias': mean_cv_bias
        },
        'validation_metrics': {
            'R²': val_r2,
            'Adjusted_R²': val_adj_r2,
            'RMSE': val_rmse,
            'MSE': val_mse,
            'MAE': val_mae,
            'Bias': val_bias,
            'Feature_Importances': val_importances
        },
        'all_data_metrics': {
            'R²': all_data_r2,
            'Adjusted_R²': all_data_adj_r2,
            'RMSE': all_data_rmse,
            'MSE': all_data_mse,
            'MAE': all_data_mae,
            'Bias': all_data_bias,
            'Feature_Importances': all_data_importances
        },
        'models': {
            'validation_model': validation_model,
            'all_data_model': all_data_model
        },
        't_test_results': {
            'R²_vs_Adj_R²': {
                't_statistic': r2_vs_adj_r2_ttest.statistic,
                'p_value': r2_vs_adj_r2_ttest.pvalue
            }
        },
        'fold_actual_pred': fold_actual_pred,
        'validation_data': {
            'X_test': X_test, 
            'y_test': y_test, 
            'y_pred': val_y_pred,
            'X_test_scaled': X_test_scaled  # 添加标准化后的测试数据
        },
        'all_data_split': {
            'X_test': X_all_scaled,
            'y_test': y,
            'y_pred': all_data_y_pred
        },
        'feature_names': X.columns,
        'scaler': scaler
    }
    
    return results

# 保存模型函数
def save_models(results, model_name, OUTPUT_DIR):
    """保存所有折的模型、验证模型和全数据模型
    
    参数:
        results: 模型训练评估结果字典
        model_name: 模型名称
        OUTPUT_DIR: 输出目录
    """
    print(f"正在保存{model_name}的所有模型和相关数据...")
    
    # 创建模型保存目录
    models_dir = os.path.join(OUTPUT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. 保存每个折叠的模型
    for fold_name, model in results['fold_models'].items():
        model_path = os.path.join(models_dir, f"{model_name}_{fold_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"已保存{fold_name}模型到: {model_path}")
    
    # 2. 保存验证模型
    validation_model_path = os.path.join(models_dir, f"{model_name}_validation_model.pkl")
    with open(validation_model_path, 'wb') as f:
        pickle.dump(results['models']['validation_model'], f)
    print(f"已保存验证模型到: {validation_model_path}")
    
    # 3. 保存全数据模型
    all_data_model_path = os.path.join(models_dir, f"{model_name}_all_data_model.pkl")
    with open(all_data_model_path, 'wb') as f:
        pickle.dump(results['models']['all_data_model'], f)
    print(f"已保存全数据模型到: {all_data_model_path}")
    
    # 4. 保存标准化器，用于未来预测时对新数据进行处理
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(results['scaler'], f)
    print(f"已保存标准化器到: {scaler_path}")
    
    # 5. 保存特征名称列表，用于确保未来预测使用相同的特征顺序
    feature_names_path = os.path.join(models_dir, f"{model_name}_feature_names.pkl")
    with open(feature_names_path, 'wb') as f:
        pickle.dump(results['feature_names'], f)
    print(f"已保存特征名称列表到: {feature_names_path}")
    
    # 6. 保存测试数据，用于后续的SHAP分析
    test_data = {
        'X_test': results['validation_data']['X_test'],
        'y_test': results['validation_data']['y_test']
    }
    
    test_data_path = os.path.join(models_dir, f"{model_name}_test_data.pkl")
    with open(test_data_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"已保存测试数据到: {test_data_path}")
    
    # 创建模型使用说明
    model_readme_path = os.path.join(models_dir, f"{model_name}_README.txt")
    with open(model_readme_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_name}模型使用说明\n")
        f.write("======================\n\n")
        f.write("目录内容：\n")
        f.write(f"1. 交叉验证模型：{model_name}_fold_X_model.pkl (X=1-5)\n")
        f.write(f"2. 验证模型：{model_name}_validation_model.pkl\n")
        f.write(f"3. 全数据模型：{model_name}_all_data_model.pkl\n")
        f.write(f"4. 标准化器：{model_name}_scaler.pkl\n")
        f.write(f"5. 特征名称：{model_name}_feature_names.pkl\n\n")
        
        f.write("模型加载和使用示例：\n")
        f.write("```python\n")
        f.write("import pickle\n")
        f.write("import pandas as pd\n")
        f.write("import numpy as np\n\n")
        
        f.write(f"# 加载模型（以全数据模型为例）\n")
        f.write(f"with open('{model_name}_all_data_model.pkl', 'rb') as f:\n")
        f.write("    model = pickle.load(f)\n\n")
        
        f.write(f"# 加载标准化器\n")
        f.write(f"with open('{model_name}_scaler.pkl', 'rb') as f:\n")
        f.write("    scaler = pickle.load(f)\n\n")
        
        f.write(f"# 加载特征名称\n")
        f.write(f"with open('{model_name}_feature_names.pkl', 'rb') as f:\n")
        f.write("    feature_names = pickle.load(f)\n\n")
        
        f.write("# 准备新数据 (假设new_data是一个包含所需特征的DataFrame)\n")
        f.write("# 确保数据包含所有必要的特征列\n")
        f.write("X_new = new_data[feature_names]\n\n")
        
        f.write("# 标准化数据\n")
        f.write("X_new_scaled = scaler.transform(X_new)\n\n")
        
        f.write("# 使用模型进行预测\n")
        f.write("predictions = model.predict(X_new_scaled)\n")
        f.write("```\n\n")
        
        f.write("注意事项：\n")
        f.write("- 全数据模型通常是最合适用于未来预测的模型，因为它使用了所有可用数据进行训练\n")
        f.write("- 确保新数据包含训练模型时使用的所有特征，并以相同的顺序提供\n")
        f.write("- 使用提供的标准化器对新数据进行预处理，以确保与训练数据相同的标准化\n")
    
    print(f"已保存模型使用说明到: {model_readme_path}")
    return models_dir

# 可视化模型结果
def visualize_model_results(results, model_name, cn_model_name, OUTPUT_DIR):
    """为模型评估创建综合可视化"""
    print(f"正在为{cn_model_name}创建可视化结果...")
    
    # 创建可视化存储目录
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    metrics_dir = os.path.join(OUTPUT_DIR, 'metrics')
    importance_dir = os.path.join(OUTPUT_DIR, 'importance')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(importance_dir, exist_ok=True)
    
    # 获取折叠度量数据
    fold_metrics_df = results['fold_metrics']
    
    # 1. 模型比较：R²与调整后R²的对比
    metrics_data = {
        '模型': [f'折叠 {i}' for i in range(1, len(fold_metrics_df)+1)] + ['平均CV', '验证集', '全数据'],
        'R²': list(fold_metrics_df['R²']) + [
            results['mean_cv_metrics']['R²'],
            results['validation_metrics']['R²'],
            results['all_data_metrics']['R²']
        ],
        '调整后R²': list(fold_metrics_df['Adjusted_R²']) + [
            results['mean_cv_metrics']['Adjusted_R²'],
            results['validation_metrics']['Adjusted_R²'],
            results['all_data_metrics']['Adjusted_R²']
        ]
    }
    metrics_comparison = pd.DataFrame(metrics_data)
    
    # 重塑数据用于seaborn可视化
    metrics_melted = pd.melt(metrics_comparison, id_vars=['模型'], 
                          value_vars=['R²', '调整后R²'],
                          var_name='指标类型', value_name='数值')
    
    # 绘图
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='模型', y='数值', hue='指标类型', data=metrics_melted)
    plt.title(f'{cn_model_name} - R²与调整后R²比较', fontsize=16)
    plt.ylabel('数值', fontsize=14)
    plt.xlabel('模型', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_r2_comparison.png'), dpi=600)
    plt.close()
    
    # 保存指标为CSV
    metrics_comparison.to_csv(os.path.join(metrics_dir, f'{model_name}_r2_metrics.csv'), index=False)
    
    # 2. 折叠指标比较
    plt.figure(figsize=(14, 10))
    metrics_to_plot = [('R²', 'R²'), ('Adjusted_R²', '调整后R²'), 
                       ('RMSE', 'RMSE'), ('MSE', 'MSE'), 
                       ('MAE', 'MAE'), ('Bias', '偏差')]
    
    for i, (metric, cn_metric) in enumerate(metrics_to_plot, 1):
        plt.subplot(3, 2, i)
        fold_values = fold_metrics_df[metric].tolist()
        mean_cv_value = results['mean_cv_metrics'][metric]
        validation_value = results['validation_metrics'][metric]
        all_data_value = results['all_data_metrics'][metric]
        
        bars = plt.bar(
            ['折叠 1', '折叠 2', '折叠 3', '折叠 4', '折叠 5', '平均CV', '验证集', '全数据'],
            fold_values + [mean_cv_value, validation_value, all_data_value]
        )
        
        # 为不同类型的模型使用不同颜色
        bars[5].set_color('green')    # 平均CV
        bars[6].set_color('orange')   # 验证集
        bars[7].set_color('red')      # 全数据
        
        plt.title(f'{cn_metric}')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_all_metrics.png'), dpi=600)
    plt.close()
    
    # 3. 特征重要性可视化并保存到文件
    # 为每个折叠绘制特征重要性
    for i, fold_metrics in enumerate(results['fold_metrics'].to_dict('records')):
        fold_num = i + 1
        feature_importances = fold_metrics['Feature_Importances']
        
        # 保存完整特征重要性到CSV文件
        feature_importances.to_csv(
            os.path.join(importance_dir, f'{model_name}_fold{fold_num}_importance.csv'),
            index=False
        )
        
        # 绘制前20个重要特征
        plt.figure(figsize=(12, 8))
        top_features = feature_importances.head(80)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'{cn_model_name} 折叠 {fold_num} 的前20个重要特征')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_fold{fold_num}_top20_importance.png'), dpi=600)
        plt.close()
    
    # 为验证模型绘制特征重要性
    val_importances = results['validation_metrics']['Feature_Importances']
    val_importances.to_csv(
        os.path.join(importance_dir, f'{model_name}_validation_importance.csv'),
        index=False
    )
    
    plt.figure(figsize=(12, 8))
    top_val_features = val_importances.head(80)
    sns.barplot(x='Importance', y='Feature', data=top_val_features)
    plt.title(f'{cn_model_name} 验证模型的前20个重要特征')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_validation_top20_importance.png'), dpi=600)
    plt.close()
    
    # 为全数据模型绘制特征重要性
    all_data_importances = results['all_data_metrics']['Feature_Importances']
    all_data_importances.to_csv(
        os.path.join(importance_dir, f'{model_name}_all_data_importance.csv'),
        index=False
    )
    
    plt.figure(figsize=(12, 8))
    top_all_data_features = all_data_importances.head(80)
    sns.barplot(x='Importance', y='Feature', data=top_all_data_features)
    plt.title(f'{cn_model_name} 全数据模型的前20个重要特征')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_all_data_top20_importance.png'), dpi=600)
    plt.close()
    
    # 4. 为每种模型类型创建高级残差图
    model_types = {
        'validation': {
            'name': '验证',
            'X_test': results['validation_data']['X_test'],
            'y_test': results['validation_data']['y_test'],
            'y_pred': results['validation_data']['y_pred']
        },
        'all_data': {
            'name': '全数据',
            'X_test': results['all_data_split']['X_test'],
            'y_test': results['all_data_split']['y_test'],
            'y_pred': results['all_data_split']['y_pred'] 
        }
    }
    
    for model_type, data in model_types.items():
        # 获取数据
        y_test = data['y_test']
        y_pred = data['y_pred']
        residuals = y_test - y_pred
        cn_model_type = data['name']
        
        # A. 实际值与预测值对比图
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.title(f'{cn_model_name} {cn_model_type}模型 - 实际值与预测值对比')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.grid(True)
        # 添加R²和RMSE注释
        if model_type == 'validation':
            r2 = results['validation_metrics']['R²']
            adj_r2 = results['validation_metrics']['Adjusted_R²']
            rmse = results['validation_metrics']['RMSE']
        else:
            r2 = results['all_data_metrics']['R²']
            adj_r2 = results['all_data_metrics']['Adjusted_R²']
            rmse = results['all_data_metrics']['RMSE']
        
        plt.annotate(
            f"R² = {r2:.4f}\n调整后R² = {adj_r2:.4f}\nRMSE = {rmse:.4f}", 
            xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_{model_type}_actual_vs_predicted.png'), dpi=600)
        plt.close()
        
        # B. 残差与预测值对比图
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'{cn_model_name} {cn_model_type}模型 - 残差与预测值对比')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_{model_type}_residuals_vs_predicted.png'), dpi=600)
        plt.close()
        
        # C. 残差QQ图
        plt.figure(figsize=(10, 8))
        spstats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{cn_model_name} {cn_model_type}模型 - 残差QQ图')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_{model_type}_residuals_qq.png'), dpi=600)
        plt.close()
        
        # D. 残差序列图（与索引对比）
        plt.figure(figsize=(10, 8))
        plt.plot(range(len(residuals)), residuals, 'o-', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'{cn_model_name} {cn_model_type}模型 - 残差序列图')
        plt.xlabel('观测索引')
        plt.ylabel('残差')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_{model_type}_residual_sequence.png'), dpi=600)
        plt.close()
        
        # E. 残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'{cn_model_name} {cn_model_type}模型 - 残差直方图')
        plt.xlabel('残差值')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_{model_type}_residuals_histogram.png'), dpi=600)
        plt.close()

# 执行配对t检验
# 执行配对t检验
def perform_paired_ttests(all_results):
    """对不同模型间的指标进行配对t检验并创建表格"""
    print("正在执行模型间的配对t检验...")
    
    # 提取每个模型的结果
    models = list(all_results.keys())
    model_cn_names = {k: FEATURE_GROUPS[k]['name'] for k in models}
    
    # 要比较的指标
    metrics = ['R²', 'Adjusted_R²', 'RMSE']
    cn_metrics = {'R²': 'R2', 'Adjusted_R²': 'Adjusted R2', 'RMSE': 'RMSE'}
    
    # 存储所有t检验结果
    all_ttest_results = []
    
    # 对每对模型进行比较
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # 避免重复比较
                # 对每个指标进行t检验
                for metric in metrics:
                    # 获取两个模型的该指标在各折上的值
                    metric_model1 = all_results[model1]['fold_metrics'][metric].values
                    metric_model2 = all_results[model2]['fold_metrics'][metric].values
                    
                    # 进行配对t检验
                    t_stat, p_val = stats.ttest_rel(metric_model1, metric_model2)
                    
                    # 存储结果
                    all_ttest_results.append({
                        'Model_1': model_cn_names[model1],
                        'Model_2': model_cn_names[model2],
                        'Metric': cn_metrics[metric],
                        'P_value': p_val
                    })
    
    # 创建结果DataFrame
    ttest_df = pd.DataFrame(all_ttest_results)
    
    # 保存为CSV文件
    ttest_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison', 'model_ttest_comparison.csv'), index=False)
    
    # 透视表形式，更易于查看
    pivot_ttest = ttest_df.pivot_table(
        index=['Model_1', 'Model_2'], 
        columns='Metric', 
        values='P_value'
    ).reset_index()
    
    # 保存透视表形式
    pivot_ttest.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison', 'model_ttest_pivot.csv'), index=False)
    
    print(f"t检验结果已保存至 model_comparison 目录")
    
    return {
        'ttest_results': ttest_df,
        'ttest_pivot': pivot_ttest
    }

# 模型汇总比较
def summarize_models(all_results):
    """创建所有模型的汇总比较"""
    print("正在创建模型汇总比较...")
    
    models_summary = []
    cn_model_names = {k: v['name'] for k, v in FEATURE_GROUPS.items()}
    
    for model_name, results in all_results.items():
        cn_name = cn_model_names[model_name]
        
        # 提取验证指标
        validation_metrics = results['validation_metrics'].copy()
        validation_metrics['模型'] = f"{cn_name}_验证集"
        
        # 提取全数据指标
        all_data_metrics = results['all_data_metrics'].copy()
        all_data_metrics['模型'] = f"{cn_name}_全数据"
        
        # 提取平均CV指标
        mean_cv_metrics = results['mean_cv_metrics'].copy()
        mean_cv_metrics['模型'] = f"{cn_name}_平均CV"
        
        # 添加到列表
        models_summary.append(validation_metrics)
        models_summary.append(all_data_metrics)
        models_summary.append(mean_cv_metrics)
    
    # 创建DataFrame
    summary_df = pd.DataFrame(models_summary)
    
    # 如果存在Feature_Importances列，则移除
    if 'Feature_Importances' in summary_df.columns:
        summary_df = summary_df.drop('Feature_Importances', axis=1)
    
    # 重新排序列，使Model列在最前面
    cols = ['模型'] + [col for col in summary_df.columns if col != '模型']
    summary_df = summary_df[cols]
    
    # 保存为CSV文件
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison', 'model_summary.csv'), index=False)
    
    # 绘制关键指标的比较图
    metrics_to_plot = [('R²', 'R²'), ('Adjusted_R²', '调整后R²'), 
                      ('RMSE', 'RMSE'), ('MSE', 'MSE'), 
                      ('MAE', 'MAE'), ('Bias', '偏差')]
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, n_metrics * 5))
    
    for i, (metric, cn_metric) in enumerate(metrics_to_plot):
        ax = axes[i]
        sns.barplot(x='模型', y=metric, data=summary_df, ax=ax)
        ax.set_title(f'{cn_metric}比较', fontsize=14)
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel(cn_metric, fontsize=12)
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison', 'models_comparison_all_metrics.png'), dpi=600)
    plt.close()
    
    # 创建facetgrid视图以获得更紧凑的展示
    plt.figure(figsize=(20, 15))
    
    # 重塑数据以便facetgrid使用
    plot_data = pd.melt(summary_df, id_vars=['模型'], value_vars=[m for m, _ in metrics_to_plot],
                     var_name='指标', value_name='数值')
    
    # 创建facetgrid
    g = sns.FacetGrid(plot_data, col='指标', col_wrap=3, height=5, sharey=False)
    g.map_dataframe(sns.barplot, x='模型', y='数值')
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=90)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison', 'models_comparison_facet.png'), dpi=600)
    plt.close()
    
    return summary_df

# 创建README文件
def create_readme_file():
    """创建带有调整后R²说明的README文件"""
    with open(os.path.join(OUTPUT_DIR, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write("随机森林模型评估结果\n")
        f.write("====================\n\n")
        f.write("本目录包含使用不同变量集构建的随机森林模型的结果。\n\n")
        
        f.write("什么是调整后R²（Adjusted R-squared）？\n")
        f.write("--------------------------------\n")
        f.write("调整后R²对使用过多变量的模型进行惩罚。\n\n")
        f.write("公式：调整后R² = 1 - ((1 - R²) * (n - 1) / (n - p - 1))\n\n")
        f.write("其中：\n")
        f.write("n = 样本数量\n")
        f.write("p = 特征（变量）数量\n")
        f.write("R² = 原始R²值\n\n")
        
        f.write("调整后R²有助于确定添加变量是否真正改善了模型，或者模型是否过度拟合。\n")
        f.write("更高的调整后R²表明模型更好且复杂度适当。\n\n")
        
        f.write("目录结构：\n")
        f.write("--------\n")
        f.write("- /climate_vars/ - 气候变量模型的结果\n")
        f.write("- /all_vars/ - 全部变量模型的结果\n")
        f.write("- /other_vars/ - 其他变量模型的结果\n")
        f.write("- /selected_vars/ - 筛选变量模型的结果\n")
        f.write("- /model_comparison/ - 所有模型的比较\n\n")
        
        f.write("每个模型目录包含：\n")
        f.write("- /plots/ - 模型性能的可视化图表\n")
        f.write("- /metrics/ - 性能指标的CSV文件\n")
        f.write("- /importance/ - 特征重要性分析\n")
        f.write("- /models/ - 保存的模型文件（包括每个折叠的模型、验证模型和全数据模型）\n")

# 显示进度信息的函数
def display_progress_info(step, total_steps, start_time):
    """显示当前进度和估计剩余时间"""
    current_time = time.time()
    elapsed_time = current_time - start_time
    progress_percent = step / total_steps * 100
    
    if step > 0:
        estimated_total_time = elapsed_time * total_steps / step
        estimated_remaining_time = estimated_total_time - elapsed_time
        
        print(f"进度: {progress_percent:.1f}% 完成 | 已用时间: {format_time(elapsed_time)} | 估计剩余时间: {format_time(estimated_remaining_time)}")
    else:
        print(f"进度: {progress_percent:.1f}% 完成 | 开始处理...")

# 格式化时间的辅助函数
def format_time(seconds):
    """将秒数格式化为人类可读的时间格式"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def select_representative_samples(X, n_clusters=80000):
    """
    使用KMeans聚类选择代表性样本
    
    参数:
        X: 特征矩阵
        n_clusters: 聚类数量
        
    返回:
        代表性样本的索引
    """
    print(f"使用KMeans选择{n_clusters}个代表性样本...")
    
    # 如果样本数小于聚类数，直接返回所有样本
    if X.shape[0] <= n_clusters:
        print(f"样本数({X.shape[0]})小于等于请求的聚类数({n_clusters})，返回所有样本")
        return np.arange(X.shape[0])
    
    # 使用KMeans聚类
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=27)
    kmeans.fit(X)
    
    # 找到最接近每个聚类中心的样本点
    closest_points, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    
    print(f"已选择{len(closest_points)}个代表性样本")
    return closest_points

def generate_yearly_rf_shap(shap_values, X_sample, sample_indices, year_info, 
                          feature_names, geo_info, OUTPUT_DIR):
    """
    生成按年份分组的随机森林SHAP输出，考虑对子变量
    
    参数:
        shap_values: SHAP值数组
        X_sample: 样本特征
        sample_indices: 样本索引
        year_info: 年份信息字典
        feature_names: 特征名称
        geo_info: 几何信息字典
        OUTPUT_DIR: 输出目录
    """
    # 创建年度SHAP目录
    yearly_dir = os.path.join(OUTPUT_DIR, "yearly_shap")
    os.makedirs(yearly_dir, exist_ok=True)
    
    print("生成年度SHAP输出...")
    
    # 创建样本ID、年份和SHAP值的数据框
    sample_data = pd.DataFrame()
    
    # 添加基本信息
    if year_info is not None and not year_info.empty:
        sample_data['ID'] = year_info.loc[sample_indices, 'ID'].values
        sample_data['year'] = year_info.loc[sample_indices, 'year'].values
    else:
        sample_data['ID'] = np.arange(len(sample_indices))
        sample_data['year'] = 0  # 默认年份
    
    # 添加几何信息
    if geo_info is not None:
        sample_data['geometry'] = geo_info.loc[sample_indices, 'geometry'].values
    
    # 添加SHAP值
    for i, feature in enumerate(feature_names):
        if i < shap_values.shape[1]:
            sample_data[feature] = shap_values[:, i]
    
    # 按年份分组处理
    for year, group in sample_data.groupby('year'):
        if year == 0:
            continue  # 跳过默认年份
            
        print(f"处理{year}年的SHAP值...")
        
        # 创建年份目录
        year_dir = os.path.join(yearly_dir, f"year_{year}")
        os.makedirs(year_dir, exist_ok=True)
        
        # 确保geometry列是几何对象
        if 'geometry' in group.columns:
            try:
                if not isinstance(group['geometry'].iloc[0], (gpd.geoseries.GeoSeries, gpd.geodataframe.GeoDataFrame)):
                    group['geometry'] = group['geometry'].apply(
                        lambda x: wkt.loads(x) if isinstance(x, str) else x
                    )
                gdf = gpd.GeoDataFrame(group, geometry='geometry')
            except Exception as e:
                print(f"转换几何列出错: {e}")
                gdf = pd.DataFrame(group)
        else:
            gdf = pd.DataFrame(group)
        
        # ==== 创建原始SHAP值数据 ====
        # 计算total列 (所有特征的综合影响)
        # 处理非对子变量
        non_pair_abs_sum = np.zeros(len(gdf))
        valid_non_pairs = [f for f in NON_PAIR_FEATURES if f in gdf.columns]
        for col in valid_non_pairs:
            non_pair_abs_sum += np.abs(gdf[col].values)
        
        # 处理对子变量
        pair_abs_mean_sum = np.zeros(len(gdf))
        for base, derived in ALL_PAIRS:
            if base in gdf.columns and derived in gdf.columns:
                pair_abs = (np.abs(gdf[base].values) + np.abs(gdf[derived].values)) / 2
                pair_abs_mean_sum += pair_abs
        
        # 添加total列
        gdf['total'] = non_pair_abs_sum + pair_abs_mean_sum
        
        # 保存原始SHAP值CSV
        orig_csv_path = os.path.join(year_dir, f"shap_original_{year}.csv")
        gdf.to_csv(orig_csv_path, index=False)
        print(f"✓ {year}年原始SHAP值已保存为CSV: {orig_csv_path}")
        
        # 分割保存原始SHAP值SHP（因为字段超过255）
        orig_shp_base = os.path.join(year_dir, f"shap_original_{year}")
        try:
            split_and_save_shapefile(gdf, orig_shp_base, max_fields=250, suffix_format="_{}") 
            print(f"✓ {year}年原始SHAP值已保存为SHP: {orig_shp_base}_1.shp, {orig_shp_base}_2.shp")
        except Exception as e:
            print(f"保存原始SHAP值SHP出错: {e}")
        
        # ==== 创建统计值数据 ====
        stats_data = {
            'ID': gdf['ID'].values,
            'year': gdf['year'].values
        }
        
        if 'geometry' in gdf.columns:
            stats_data['geometry'] = gdf['geometry'].values
        
        # 计算ma和ms值
        ma_values = {}
        ms_values = {}
        
        for base, derived in ALL_PAIRS:
            if base in gdf.columns and derived in gdf.columns:
                # 方向值 (ma)
                ma_values[f"{base}ma"] = (gdf[base] + gdf[derived]) / 2
                
                # 强度值 (ms)
                ms_values[f"{base}ms"] = (np.abs(gdf[base]) + np.abs(gdf[derived])) / 2
        
        # 创建统计值DataFrame
        stats_data.update(ma_values)
        stats_data.update(ms_values)
        stats_df = pd.DataFrame(stats_data)
        
        if 'geometry' in stats_data:
            try:
                stats_gdf = gpd.GeoDataFrame(stats_df, geometry='geometry')
            except Exception as e:
                print(f"创建统计GeoDataFrame出错: {e}")
                stats_gdf = stats_df
        else:
            stats_gdf = stats_df
        
        # 计算totalma列
        non_pair_sum = np.zeros(len(stats_gdf))
        if valid_non_pairs:
            for col in valid_non_pairs:
                non_pair_sum += gdf[col].values
        
        ma_sum = np.zeros(len(stats_gdf))
        for key in ma_values.keys():
            ma_sum += stats_gdf[key].values
        
        # 添加totalma列
        stats_gdf['totalma'] = non_pair_sum + ma_sum
        
        # 保存统计值CSV
        stats_csv_path = os.path.join(year_dir, f"shap_stats_{year}.csv")
        stats_gdf.to_csv(stats_csv_path, index=False)
        print(f"✓ {year}年统计SHAP已保存为CSV: {stats_csv_path}")
        
        # 分割保存统计值SHP（因为字段超过255）
        stats_shp_base = os.path.join(year_dir, f"shap_stats_{year}")
        try:
            split_and_save_shapefile(stats_gdf, stats_shp_base, max_fields=250, suffix_format="_{}")
            print(f"✓ {year}年统计SHAP已保存为SHP: {stats_shp_base}_1.shp, {stats_shp_base}_2.shp")
        except Exception as e:
            print(f"保存统计SHAP值SHP出错: {e}")
    
    print("所有年份SHAP输出已完成")

def split_and_save_shapefile(gdf, base_path, max_fields=250, suffix_format="_{}", base_fields=None):
    """
    将大型GeoDataFrame分割成多个Shapefile
    
    参数:
        gdf: GeoDataFrame对象
        base_path: 基础路径（不含扩展名）
        max_fields: 每个shapefile的最大字段数
        suffix_format: 后缀格式，例如"_{}"将生成_1, _2形式的后缀
        base_fields: 每个分割文件中都需要包含的基础字段
    """
    if base_fields is None:
        base_fields = ['ID', 'geometry']
        if 'year' in gdf.columns:
            base_fields.append('year')
    
    # 确保基础字段在GDF中存在
    base_fields = [f for f in base_fields if f in gdf.columns]
    
    # 其他字段
    other_cols = [col for col in gdf.columns if col not in base_fields]
    
    # 计算需要分几个部分
    num_parts = max(1, (len(other_cols) + max_fields - len(base_fields) - 1) // (max_fields - len(base_fields)))
    
    # 当只需要一个部分时
    if num_parts == 1:
        part_path = f"{base_path}.shp"
        try:
            gdf.to_file(part_path, driver="ESRI Shapefile", encoding='UTF-8')
            print(f"✓ 保存为单个SHP文件: {part_path}")
        except Exception as e:
            print(f"保存SHP文件失败: {e}")
        return
    
    # 分批处理字段
    for i in range(num_parts):
        start_idx = i * (max_fields - len(base_fields))
        end_idx = min((i+1) * (max_fields - len(base_fields)), len(other_cols))
        part_cols = base_fields + other_cols[start_idx:end_idx]
        part_df = gdf[part_cols].copy()
        part_path = f"{base_path}{suffix_format.format(i+1)}.shp"
        
        try:
            if 'geometry' in part_df.columns and isinstance(part_df, pd.DataFrame):
                part_gdf = gpd.GeoDataFrame(part_df, geometry='geometry')
                part_gdf.to_file(part_path, driver="ESRI Shapefile", encoding='UTF-8')
            else:
                part_df.to_file(part_path, driver="ESRI Shapefile", encoding='UTF-8')
            print(f"✓ 部分{i+1}/{num_parts}保存为: {part_path}")
        except Exception as e:
            print(f"✗ 部分{i+1}/{num_parts}保存失败: {part_path}, 错误: {e}")

def generate_aggregated_rf_shap(yearly_dir, OUTPUT_DIR, feature_names):
    """
    基于年度SHAP文件生成聚合SHAP文件
    
    参数:
        yearly_dir: 年度SHAP文件目录
        OUTPUT_DIR: 输出目录
        feature_names: 特征名称列表
    """
    # 创建聚合目录
    agg_dir = os.path.join(OUTPUT_DIR, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    
    print("生成聚合SHAP输出...")
    
    # 获取所有年份目录
    year_dirs = [os.path.join(yearly_dir, d) for d in os.listdir(yearly_dir) 
                if os.path.isdir(os.path.join(yearly_dir, d)) and d.startswith('year_')]
    
    if not year_dirs:
        print("⚠️ 未找到年度SHAP目录，无法聚合")
        return
    
    # 准备收集所有年份数据的容器
    all_years_original = []  # 用于收集原始SHAP值
    all_years_stats = []     # 用于收集统计值
    
    for year_dir in year_dirs:
        year = os.path.basename(year_dir).replace('year_', '')
        
        # 1. 尝试加载原始SHAP值CSV
        orig_csv = os.path.join(year_dir, f"shap_original_{year}.csv")
        if os.path.exists(orig_csv):
            try:
                orig_df = pd.read_csv(orig_csv)
                if 'geometry' in orig_df.columns:
                    orig_df['geometry'] = orig_df['geometry'].apply(
                        lambda x: wkt.loads(x) if isinstance(x, str) else x
                    )
                    orig_df = gpd.GeoDataFrame(orig_df, geometry='geometry')
                all_years_original.append(orig_df)
                print(f"✓ 已加载{year}年原始SHAP值CSV")
            except Exception as e:
                print(f"✗ 加载{year}年原始SHAP值CSV出错: {e}")
                
                # 2. 如果CSV加载失败，尝试加载SHP文件
                orig_shp_parts = glob.glob(os.path.join(year_dir, f"shap_original_{year}_*.shp"))
                if orig_shp_parts:
                    try:
                        # 加载并合并所有部分
                        orig_parts_data = []
                        for shp in orig_shp_parts:
                            part_gdf = gpd.read_file(shp)
                            orig_parts_data.append(part_gdf)
                        
                        # 合并多个部分，确保不重复基础字段
                        base_cols = ['ID', 'geometry']
                        if 'year' in orig_parts_data[0].columns:
                            base_cols.append('year')
                        
                        merged_orig = orig_parts_data[0]
                        for part in orig_parts_data[1:]:
                            # 排除基础字段，只添加额外字段
                            extra_cols = [col for col in part.columns if col not in base_cols]
                            for col in extra_cols:
                                merged_orig[col] = part[col]
                        
                        all_years_original.append(merged_orig)
                        print(f"✓ 已加载{year}年原始SHAP值SHP文件")
                    except Exception as e2:
                        print(f"✗ 加载{year}年原始SHAP值SHP出错: {e2}")
        
        # 3. 加载统计值CSV
        stats_csv = os.path.join(year_dir, f"shap_stats_{year}.csv")
        if os.path.exists(stats_csv):
            try:
                stats_df = pd.read_csv(stats_csv)
                if 'geometry' in stats_df.columns:
                    stats_df['geometry'] = stats_df['geometry'].apply(
                        lambda x: wkt.loads(x) if isinstance(x, str) else x
                    )
                    stats_df = gpd.GeoDataFrame(stats_df, geometry='geometry')
                all_years_stats.append(stats_df)
                print(f"✓ 已加载{year}年统计值CSV")
            except Exception as e:
                print(f"✗ 加载{year}年统计值CSV出错: {e}")
                
                # 4. 如果CSV加载失败，尝试加载SHP文件
                stats_shp_parts = glob.glob(os.path.join(year_dir, f"shap_stats_{year}_*.shp"))
                if stats_shp_parts:
                    try:
                        # 加载并合并所有部分
                        stats_parts_data = []
                        for shp in stats_shp_parts:
                            part_gdf = gpd.read_file(shp)
                            stats_parts_data.append(part_gdf)
                        
                        # 合并多个部分，确保不重复基础字段
                        base_cols = ['ID', 'geometry']
                        if 'year' in stats_parts_data[0].columns:
                            base_cols.append('year')
                        
                        merged_stats = stats_parts_data[0]
                        for part in stats_parts_data[1:]:
                            # 排除基础字段，只添加额外字段
                            extra_cols = [col for col in part.columns if col not in base_cols]
                            for col in extra_cols:
                                merged_stats[col] = part[col]
                        
                        all_years_stats.append(merged_stats)
                        print(f"✓ 已加载{year}年统计值SHP文件")
                    except Exception as e2:
                        print(f"✗ 加载{year}年统计值SHP出错: {e2}")
    
    if not all_years_original or not all_years_stats:
        print("⚠️ 未能成功加载任何年度SHAP数据")
        return
    
    # 合并所有年份数据
    all_orig_data = pd.concat(all_years_original, ignore_index=True)
    all_stats_data = pd.concat(all_years_stats, ignore_index=True)
    
    # 按ID分组聚合
    unique_ids = all_orig_data['ID'].unique()
    
    # 准备聚合结果
    direction_rows = []  # 方向聚合 (ma)
    strength_rows = []   # 强度聚合 (ms)
    
    for unique_id in unique_ids:
        # 获取该ID的所有年份数据
        id_orig = all_orig_data[all_orig_data['ID'] == unique_id]
        id_stats = all_stats_data[all_stats_data['ID'] == unique_id]
        
        if len(id_orig) == 0 or len(id_stats) == 0:
            continue
        
        # 获取几何形状
        id_geometry = id_orig.iloc[0]['geometry'] if 'geometry' in id_orig.columns else None
        
        # 1. 方向聚合
        dir_row = {'ID': unique_id}
        if id_geometry is not None:
            dir_row['geometry'] = id_geometry
        
        # 添加非对子变量（原始值，表示方向）
        valid_non_pairs = [f for f in NON_PAIR_FEATURES if f in id_orig.columns]
        for col in valid_non_pairs:
            dir_row[col] = id_orig[col].mean()
        
        # 添加ma值（方向）
        ma_cols = [col for col in id_stats.columns if col.endswith('ma') and col != 'totalma']
        for col in ma_cols:
            dir_row[col] = id_stats[col].mean()
        
        # 添加totalma（总方向）
        if 'totalma' in id_stats.columns:
            dir_row['totalma'] = id_stats['totalma'].mean()
            
        direction_rows.append(dir_row)
        
        # 2. 强度聚合
        str_row = {'ID': unique_id}
        if id_geometry is not None:
            str_row['geometry'] = id_geometry
        
        # 添加非对子变量（绝对值，表示强度）
        for col in valid_non_pairs:
            str_row[col] = id_orig[col].abs().mean()
        
        # 添加ms值（强度）
        ms_cols = [col for col in id_stats.columns if col.endswith('ms')]
        for col in ms_cols:
            str_row[col] = id_stats[col].mean()
        
        # 添加total（总强度）
        if 'total' in id_orig.columns:
            str_row['total'] = id_orig['total'].mean()
            
        strength_rows.append(str_row)
    
    # 创建GeoDataFrame
    if direction_rows and 'geometry' in direction_rows[0]:
        direction_gdf = gpd.GeoDataFrame(direction_rows, geometry='geometry')
        strength_gdf = gpd.GeoDataFrame(strength_rows, geometry='geometry')
    else:
        direction_gdf = pd.DataFrame(direction_rows)
        strength_gdf = pd.DataFrame(strength_rows)
    
    # 保存CSV
    dir_csv_path = os.path.join(agg_dir, "shap_direction_aggregated.csv")
    str_csv_path = os.path.join(agg_dir, "shap_strength_aggregated.csv")
    
    direction_gdf.to_csv(dir_csv_path, index=False)
    strength_gdf.to_csv(str_csv_path, index=False)
    
    print(f"✓ 方向聚合CSV已保存: {dir_csv_path}")
    print(f"✓ 强度聚合CSV已保存: {str_csv_path}")
    
    # 保存SHP文件
    dir_shp_base = os.path.join(agg_dir, "shap_direction_aggregated")
    str_shp_base = os.path.join(agg_dir, "shap_strength_aggregated")
    
    try:
        split_and_save_shapefile(direction_gdf, dir_shp_base, max_fields=250)
        print(f"✓ 方向聚合SHP已保存: {dir_shp_base}.shp")
    except Exception as e:
        print(f"✗ 方向聚合SHP保存失败: {e}")
    
    try:
        split_and_save_shapefile(strength_gdf, str_shp_base, max_fields=250)
        print(f"✓ 强度聚合SHP已保存: {str_shp_base}.shp")
    except Exception as e:
        print(f"✗ 强度聚合SHP保存失败: {e}")

def calculate_climate_stats(shap_values, sample_indices, year_info, feature_names, geo_info, OUTPUT_DIR):
    """
    计算气候变量统计信息并保存
    
    参数:
        shap_values: SHAP值数组
        sample_indices: 样本索引
        year_info: 年份信息
        feature_names: 特征名称
        geo_info: 几何信息
        OUTPUT_DIR: 输出目录
    """
    # 创建气候变量统计目录
    climate_dir = os.path.join(OUTPUT_DIR, "climate_stats")
    os.makedirs(climate_dir, exist_ok=True)
    
    print("计算气候变量统计信息...")
    
    # 创建基础数据框
    climate_data = pd.DataFrame()
    
    # 添加基本信息
    if year_info is not None and not year_info.empty:
        climate_data['ID'] = year_info.loc[sample_indices, 'ID'].values
        climate_data['year'] = year_info.loc[sample_indices, 'year'].values
    else:
        climate_data['ID'] = np.arange(len(sample_indices))
        climate_data['year'] = 0
    
    # 添加几何信息
    if geo_info is not None:
        climate_data['geometry'] = geo_info.loc[sample_indices, 'geometry'].values
    
    # 添加SHAP值
    for i, feature in enumerate(feature_names):
        if i < shap_values.shape[1]:
            climate_data[feature] = shap_values[:, i]
    
    # 过滤出气候变量
    climate_variables = set()
    for base, derived in CLIMATE_PAIRS:
        if base in climate_data.columns and derived in climate_data.columns:
            climate_variables.add(base)
            climate_variables.add(derived)
    
    if not climate_variables:
        print("⚠️ 未找到任何气候变量")
        return None
    
    print(f"找到{len(climate_variables)}个气候变量")
    
    # 确保climate_data是GeoDataFrame
    if 'geometry' in climate_data.columns:
        try:
            if not isinstance(climate_data, gpd.GeoDataFrame):
                climate_data = gpd.GeoDataFrame(climate_data, geometry='geometry')
        except Exception as e:
            print(f"转换为GeoDataFrame出错: {e}")
    
    # 计算气候变量的ma和ms值
    ma_values = {}
    ms_values = {}
    
    for base, derived in CLIMATE_PAIRS:
        if base in climate_data.columns and derived in climate_data.columns:
            # 方向值 (ma)
            ma_values[f"{base}ma"] = (climate_data[base] + climate_data[derived]) / 2
            
            # 强度值 (ms)
            ms_values[f"{base}ms"] = (np.abs(climate_data[base]) + np.abs(climate_data[derived])) / 2
    
    # 添加计算值
    for col, values in ma_values.items():
        climate_data[col] = values
    
    for col, values in ms_values.items():
        climate_data[col] = values
    
    # 计算总和列
    climate_data['matotal'] = sum(climate_data[col] for col in ma_values.keys())
    climate_data['mstotal'] = sum(climate_data[col] for col in ms_values.keys())
    
    # 保存全局气候变量CSV
    csv_path = os.path.join(climate_dir, "climate_variables_stats.csv")
    climate_data.to_csv(csv_path, index=False)
    print(f"✓ 气候变量统计CSV已保存: {csv_path}")
    
    # 保存全局气候变量SHP
    shp_base = os.path.join(climate_dir, "climate_variables_stats")
    if isinstance(climate_data, gpd.GeoDataFrame):
        try:
            split_and_save_shapefile(climate_data, shp_base, max_fields=250)
            print(f"✓ 气候变量统计SHP已保存: {shp_base}.shp")
        except Exception as e:
            print(f"✗ 气候变量统计SHP保存失败: {e}")
    
    # 按年份处理
    for year, year_group in climate_data.groupby('year'):
        if year == 0:
            continue  # 跳过默认年份
            
        # 保存年度气候变量CSV
        year_csv_path = os.path.join(climate_dir, f"climate_stats_{year}.csv")
        year_group.to_csv(year_csv_path, index=False)
        print(f"✓ {year}年气候统计CSV已保存: {year_csv_path}")
        
        # 保存年度气候变量SHP
        year_shp_base = os.path.join(climate_dir, f"climate_stats_{year}")
        if isinstance(year_group, gpd.GeoDataFrame):
            try:
                split_and_save_shapefile(year_group, year_shp_base, max_fields=250)
                print(f"✓ {year}年气候统计SHP已保存: {year_shp_base}.shp")
            except Exception as e:
                print(f"✗ {year}年气候统计SHP保存失败: {e}")
    
    # 计算气候变量重要性并可视化
    climate_importance_data = []
    for col in climate_variables:
        if col in climate_data.columns:
            importance = climate_data[col].abs().mean()
            climate_importance_data.append({
                'variable': col,
                'importance': importance
            })
    
    if climate_importance_data:
        climate_importance = pd.DataFrame(climate_importance_data)
        climate_importance = climate_importance.sort_values('importance', ascending=False)
        
        # 保存为CSV
        climate_importance.to_csv(os.path.join(climate_dir, "climate_importance.csv"), index=False)
        
        # 可视化
        plt.figure(figsize=(12, 8))
        plt.barh(climate_importance['variable'], climate_importance['importance'])
        plt.xlabel('平均|SHAP|值')
        plt.ylabel('气候变量')
        plt.title('气候变量重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(climate_dir, "climate_importance.png"), dpi=600)
        plt.close()
    
    return climate_data

def generate_global_rf_shap_file(shap_values, sample_indices, year_info, feature_names, geo_info, OUTPUT_DIR):
    """
    生成全局SHAP文件
    
    参数:
        shap_values: SHAP值数组
        sample_indices: 样本索引
        year_info: 年份信息
        feature_names: 特征名称
        geo_info: 几何信息
        OUTPUT_DIR: 输出目录
    """
    # 创建全局SHAP目录
    global_dir = os.path.join(OUTPUT_DIR, "global_shap")
    os.makedirs(global_dir, exist_ok=True)
    
    print("生成全局SHAP文件...")
    
    # 创建基础数据框
    global_data = pd.DataFrame()
    
    # 添加基本信息
    if year_info is not None and not year_info.empty:
        global_data['ID'] = year_info.loc[sample_indices, 'ID'].values
        global_data['year'] = year_info.loc[sample_indices, 'year'].values
        if 'ID_year' in year_info.columns:
            global_data['ID_year'] = year_info.loc[sample_indices, 'ID_year'].values
    else:
        global_data['ID'] = np.arange(len(sample_indices))
        global_data['year'] = 0
    
    # 添加几何信息
    if geo_info is not None:
        global_data['geometry'] = geo_info.loc[sample_indices, 'geometry'].values
    
    # 添加SHAP值
    for i, feature in enumerate(feature_names):
        if i < shap_values.shape[1]:
            global_data[feature] = shap_values[:, i]
    
    # 确保数据是GeoDataFrame
    if 'geometry' in global_data.columns:
        try:
            global_data = gpd.GeoDataFrame(global_data, geometry='geometry')
        except Exception as e:
            print(f"转换为GeoDataFrame出错: {e}")
    
    # 计算对子变量的ma和ms值
    ma_values = {}
    ms_values = {}
    
    for base, derived in ALL_PAIRS:
        if base in global_data.columns and derived in global_data.columns:
            # 方向值 (ma)
            ma_values[f"{base}ma"] = (global_data[base] + global_data[derived]) / 2
            
            # 强度值 (ms)
            ms_values[f"{base}ms"] = (np.abs(global_data[base]) + np.abs(global_data[derived])) / 2
    
    # 添加ma和ms值
    for col, values in ma_values.items():
        global_data[col] = values
    
    for col, values in ms_values.items():
        global_data[col] = values
    
    # 计算total和totalma列
    # 处理非对子变量
    valid_non_pairs = [f for f in NON_PAIR_FEATURES if f in global_data.columns]
    
    # 计算total列（强度）
    non_pair_abs_sum = np.zeros(len(global_data))
    for col in valid_non_pairs:
        non_pair_abs_sum += np.abs(global_data[col].values)
    
    pair_abs_mean_sum = np.zeros(len(global_data))
    for base, derived in ALL_PAIRS:
        if base in global_data.columns and derived in global_data.columns:
            pair_abs = (np.abs(global_data[base].values) + np.abs(global_data[derived].values)) / 2
            pair_abs_mean_sum += pair_abs
    
    global_data['total'] = non_pair_abs_sum + pair_abs_mean_sum
    
    # 计算totalma列（方向）
    non_pair_sum = np.zeros(len(global_data))
    for col in valid_non_pairs:
        non_pair_sum += global_data[col].values
    
    ma_sum = np.zeros(len(global_data))
    for key in ma_values.keys():
        ma_sum += global_data[key].values
    
    global_data['totalma'] = non_pair_sum + ma_sum
    
    # 保存CSV文件
    csv_path = os.path.join(global_dir, "global_all_samples_shap.csv")
    global_data.to_csv(csv_path, index=False)
    print(f"✓ 全局SHAP CSV已保存: {csv_path}")
    
    # 保存SHP文件
    shp_base = os.path.join(global_dir, "global_all_samples_shap")
    if isinstance(global_data, gpd.GeoDataFrame):
        try:
            split_and_save_shapefile(global_data, shp_base, max_fields=250)
            print(f"✓ 全局SHAP SHP已保存")
        except Exception as e:
            print(f"✗ 全局SHAP SHP保存失败: {e}")
    
    return global_data

def analyze_rf_shap_importance(shap_values, feature_names, model_name, OUTPUT_DIR):
    """
    分析SHAP重要性，考虑变量对的特性
    
    参数:
        shap_values: SHAP值数组
        feature_names: 特征名称列表
        model_name: 模型名称
        OUTPUT_DIR: 输出目录
    """
    # 创建全局重要性目录
    importance_dir = os.path.join(OUTPUT_DIR, "importance")
    os.makedirs(importance_dir, exist_ok=True)
    
    print("分析SHAP重要性...")
    
    # 计算原始SHAP绝对值均值（每个特征的重要性）
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
    
    # 创建原始特征重要性DataFrame
    original_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_abs_mean
    })
    original_importance = original_importance.sort_values('importance', ascending=True)
    
    # 保存原始特征重要性CSV
    original_importance.to_csv(os.path.join(importance_dir, f"{model_name}_original_importance.csv"), index=False)
    
    # 计算调整后的特征重要性，考虑对子关系
    adjusted_importance = []
    
    # 1. 处理非对子变量
    valid_non_pairs = [f for f in NON_PAIR_FEATURES if f in feature_names]
    for feature in valid_non_pairs:
        if feature in original_importance['feature'].values:
            feature_idx = list(feature_names).index(feature)
            importance = shap_abs_mean[feature_idx]
            adjusted_importance.append({
                'feature': feature,
                'type': 'non_pair',
                'importance': importance
            })
    
    # 2. 处理对子变量
    for base, derived in ALL_PAIRS:
        if base in feature_names and derived in feature_names:
            base_idx = list(feature_names).index(base)
            derived_idx = list(feature_names).index(derived)
            
            # 计算对子平均强度 (ms)
            pair_importance = (shap_abs_mean[base_idx] + shap_abs_mean[derived_idx]) / 2
            
            adjusted_importance.append({
                'feature': f"{base}-{derived}",
                'type': 'pair',
                'base': base,
                'derived': derived,
                'importance': pair_importance
            })
    
    # 创建调整后的重要性DataFrame
    adjusted_df = pd.DataFrame(adjusted_importance)
    if not adjusted_df.empty:
        adjusted_df = adjusted_df.sort_values('importance', ascending=True)
        
        # 保存调整后的特征重要性CSV
        adjusted_df.to_csv(os.path.join(importance_dir, f"{model_name}_adjusted_importance.csv"), index=False)
    
    # 绘制原始特征重要性（前30个）
    plt.figure(figsize=(12, 10))
    top_features = original_importance.head(30)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'{model_name} - 原始特征重要性 (前30个)')
    plt.tight_layout()
    plt.savefig(os.path.join(importance_dir, f"{model_name}_original_top30_importance.png"), dpi=600)
    plt.close()
    
    # 绘制调整后的特征重要性（前30个）
    if not adjusted_df.empty:
        plt.figure(figsize=(12, 10))
        top_adjusted = adjusted_df.head(30)
        sns.barplot(x='importance', y='feature', data=top_adjusted)
        plt.title(f'{model_name} - 调整后特征重要性 (前30个)')
        plt.tight_layout()
        plt.savefig(os.path.join(importance_dir, f"{model_name}_adjusted_top30_importance.png"), dpi=600)
        plt.close()
    
    # 分析气候变量重要性
    climate_importance = []
    
    # 识别所有气候对子变量
    for base, derived in CLIMATE_PAIRS:
        if base in feature_names and derived in feature_names:
            base_idx = list(feature_names).index(base)
            derived_idx = list(feature_names).index(derived)
            
            # 计算对子平均强度 (ms)
            pair_importance = (shap_abs_mean[base_idx] + shap_abs_mean[derived_idx]) / 2
            
            climate_importance.append({
                'feature': f"{base}-{derived}",
                'type': 'climate',
                'base': base,
                'derived': derived,
                'importance': pair_importance
            })
    
    # 创建气候变量重要性DataFrame
    climate_df = pd.DataFrame(climate_importance)
    if not climate_df.empty:
        climate_df = climate_df.sort_values('importance', ascending=True)
        
        # 保存气候变量重要性CSV
        climate_df.to_csv(os.path.join(importance_dir, f"{model_name}_climate_importance.csv"), index=False)
        
        # 绘制气候变量重要性
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=climate_df)
        plt.title(f'{model_name} - 气候变量重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(importance_dir, f"{model_name}_climate_importance.png"), dpi=600)
        plt.close()
    
    return {
        'original': original_importance,
        'adjusted': adjusted_df,
        'climate': climate_df
    }

def generate_yearly_rf_shap_importance(shap_values, sample_indices, year_info, feature_names, OUTPUT_DIR):
    """
    生成年度SHAP重要性
    
    参数:
        shap_values: SHAP值数组
        sample_indices: 样本索引
        year_info: 年份信息
        feature_names: 特征名称列表
        OUTPUT_DIR: 输出目录
    """
    # 创建年度重要性目录
    yearly_imp_dir = os.path.join(OUTPUT_DIR, "yearly_importance")
    os.makedirs(yearly_imp_dir, exist_ok=True)
    
    print("生成年度特征重要性...")
    
    if year_info is None or year_info.empty:
        print("⚠️ 缺少年份信息，无法生成年度特征重要性")
        return
    
    # 获取唯一年份
    unique_years = sorted(year_info.loc[sample_indices, 'year'].unique())
    yearly_top_features = {}
    
    for year in unique_years:
        # 创建年份目录
        year_dir = os.path.join(yearly_imp_dir, f"year_{year}")
        os.makedirs(year_dir, exist_ok=True)
        
        # 获取该年份的样本索引
        year_mask = year_info.loc[sample_indices, 'year'] == year
        year_indices = np.where(year_mask)[0]
        
        if len(year_indices) == 0:
            print(f"⚠️ {year}年没有样本")
            continue
        
        # 获取该年份的SHAP值
        year_shap = shap_values[year_indices]
        
        # 计算原始SHAP绝对值均值（每个特征的重要性）
        shap_abs_mean = np.mean(np.abs(year_shap), axis=0)
        
        # 创建原始特征重要性DataFrame
        original_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_abs_mean
        })
        original_importance = original_importance.sort_values('importance', ascending=True)
        
        # 保存原始特征重要性CSV
        original_importance.to_csv(os.path.join(year_dir, f"year_{year}_original_importance.csv"), index=False)
        
        # 计算调整后的特征重要性，考虑对子关系
        adjusted_importance = []
        
        # 1. 处理非对子变量
        valid_non_pairs = [f for f in NON_PAIR_FEATURES if f in feature_names]
        for feature in valid_non_pairs:
            if feature in original_importance['feature'].values:
                feature_idx = list(feature_names).index(feature)
                importance = shap_abs_mean[feature_idx]
                adjusted_importance.append({
                    'feature': feature,
                    'type': 'non_pair',
                    'importance': importance
                })
        
        # 2. 处理对子变量
        for base, derived in ALL_PAIRS:
            if base in feature_names and derived in feature_names:
                base_idx = list(feature_names).index(base)
                derived_idx = list(feature_names).index(derived)
                
                # 计算对子平均强度 (ms)
                pair_importance = (shap_abs_mean[base_idx] + shap_abs_mean[derived_idx]) / 2
                
                adjusted_importance.append({
                    'feature': f"{base}-{derived}",
                    'type': 'pair',
                    'base': base,
                    'derived': derived,
                    'importance': pair_importance
                })
        
        # 创建调整后的重要性DataFrame
        adjusted_df = pd.DataFrame(adjusted_importance)
        if not adjusted_df.empty:
            adjusted_df = adjusted_df.sort_values('importance', ascending=True)
            
            # 保存调整后的特征重要性CSV
            adjusted_df.to_csv(os.path.join(year_dir, f"year_{year}_adjusted_importance.csv"), index=False)
            
            # 存储top30特征用于趋势分析
            yearly_top_features[year] = adjusted_df.head(30).copy()
        
        # 绘制原始特征重要性（前30个）
        plt.figure(figsize=(12, 10))
        top_features = original_importance.head(30)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'{year}年 - 原始特征重要性 (前30个)')
        plt.tight_layout()
        plt.savefig(os.path.join(year_dir, f"year_{year}_original_top30_importance.png"), dpi=600)
        plt.close()
        
        # 绘制调整后的特征重要性（前30个）
        if not adjusted_df.empty:
            plt.figure(figsize=(12, 10))
            top_adjusted = adjusted_df.head(30)
            sns.barplot(x='importance', y='feature', data=top_adjusted)
            plt.title(f'{year}年 - 调整后特征重要性 (前30个)')
            plt.tight_layout()
            plt.savefig(os.path.join(year_dir, f"year_{year}_adjusted_top30_importance.png"), dpi=600)
            plt.close()
        
        # 分析气候变量重要性
        climate_importance = []
        
        # 识别所有气候对子变量
        for base, derived in CLIMATE_PAIRS:
            if base in feature_names and derived in feature_names:
                base_idx = list(feature_names).index(base)
                derived_idx = list(feature_names).index(derived)
                
                # 计算对子平均强度 (ms)
                pair_importance = (shap_abs_mean[base_idx] + shap_abs_mean[derived_idx]) / 2
                
                climate_importance.append({
                    'feature': f"{base}-{derived}",
                    'type': 'climate',
                    'base': base,
                    'derived': derived,
                    'importance': pair_importance
                })
        
        # 创建气候变量重要性DataFrame
        climate_df = pd.DataFrame(climate_importance)
        if not climate_df.empty:
            climate_df = climate_df.sort_values('importance', ascending=True)
            
            # 保存气候变量重要性CSV
            climate_df.to_csv(os.path.join(year_dir, f"year_{year}_climate_importance.csv"), index=False)
            
            # 绘制气候变量重要性
            plt.figure(figsize=(12, 10))
            sns.barplot(x='importance', y='feature', data=climate_df)
            plt.title(f'{year}年 - 气候变量重要性')
            plt.tight_layout()
            plt.savefig(os.path.join(year_dir, f"year_{year}_climate_importance.png"), dpi=600)
            plt.close()
    
    # 生成跨年份的特征重要性趋势
    if yearly_top_features:
        generate_importance_trends(yearly_top_features, yearly_imp_dir)

def generate_importance_trends(yearly_top_features, OUTPUT_DIR):
    """
    生成跨年份的特征重要性趋势
    
    参数:
        yearly_top_features: 各年份的top特征
        OUTPUT_DIR: 输出目录
    """
    print("生成特征重要性趋势...")
    
    # 提取所有年份中出现频率最高的特征
    all_features = {}
    for year, df in yearly_top_features.items():
        for _, row in df.iterrows():
            feature = row['feature']
            if feature not in all_features:
                all_features[feature] = 0
            all_features[feature] += 1
    
    # 获取出现频率最高的15个特征
    top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:15]
    top_feature_names = [f[0] for f in top_features]
    
    # 收集特征在各年份的重要性
    trends_data = {}
    years = sorted(yearly_top_features.keys())
    
    for feature in top_feature_names:
        trends_data[feature] = []
        for year in years:
            df = yearly_top_features[year]
            feature_row = df[df['feature'] == feature]
            if not feature_row.empty:
                trends_data[feature].append(feature_row['importance'].values[0])
            else:
                # 该年份中没有这个特征
                trends_data[feature].append(0)
    
    # 绘制趋势图
    plt.figure(figsize=(15, 10))
    for feature in top_feature_names:
        plt.plot(years, trends_data[feature], marker='o', label=feature)
    
    plt.xlabel('年份')
    plt.ylabel('平均|SHAP|值')
    plt.title('特征重要性年度趋势')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_trends.png"), dpi=600)
    plt.close()
    
    # 保存趋势数据为CSV
    trends_df = pd.DataFrame(trends_data, index=years)
    trends_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_trends.csv"))

def analyze_rf_with_shap(model, X_all_data, feature_names, OUTPUT_DIR, model_name='all_vars', 
                        sample_size=None, year_info=None, geo_info=None, batch_size=BATCH_SIZE_SHAP, n_jobs=N_JOBS):
    """
    使用SHAP分析随机森林模型，并生成完整的分析结果
    
    参数:
        model: 训练好的随机森林模型
        X_all_data: 全部数据的特征
        feature_names: 特征名称列表
        OUTPUT_DIR: 输出目录
        model_name: 模型名称
        sample_size: SHAP分析的最大样本数量 (None表示使用全部)
        year_info: 年份信息
        geo_info: 几何信息
        batch_size: 每个并行批次的样本数量
        n_jobs: 并行任务数
    """
    import gc
    
    print("开始SHAP分析...")
    
    # 创建SHAP输出目录
    shap_dir = os.path.join(OUTPUT_DIR, 'shap')
    os.makedirs(shap_dir, exist_ok=True)

    # 确定样本
    if sample_size is not None and X_all_data.shape[0] > sample_size:
        np.random.seed(27)
        sample_indices = np.random.choice(X_all_data.shape[0], sample_size, replace=False)
        X_sample = X_all_data.iloc[sample_indices] if isinstance(X_all_data, pd.DataFrame) else X_all_data[sample_indices]
        print(f"使用{len(X_sample)}个样本进行SHAP分析 (从全部{X_all_data.shape[0]}个样本中抽样)")
    else:
        X_sample = X_all_data
        sample_indices = np.arange(len(X_all_data))
        print(f"使用全部{len(X_sample)}个样本进行SHAP分析")
    
    # 创建一个全局explainer实例
    explainer = shap.TreeExplainer(model)
    
    # 使用优化后的并行计算SHAP值，限制为4个并行任务
    print(f"开始并行计算SHAP值...")
    shap_values = compute_shap_values_in_parallel(
        model, X_sample, batch_size=batch_size, n_jobs=n_jobs, OUTPUT_DIR=OUTPUT_DIR, 
        model_name=model_name  
    )
    
    # 保存SHAP值为CSV - 保留原有代码中的这一步
    print(f"保存SHAP值...")
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(os.path.join(shap_dir, f"{model_name}_shap_values.csv"), index=False)
    
    # 释放shap_df内存
    del shap_df
    gc.collect()
    
    # 创建SHAP摘要图
    print("创建SHAP摘要图...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=30)
    plt.title(f"SHAP值摘要 - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{model_name}_shap_summary.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # SHAP值排序条形图
    print("创建SHAP条形图...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=30)
    plt.title(f"SHAP值重要性 - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{model_name}_shap_importance.png"), dpi=600, bbox_inches='tight')
    plt.close()
    
    # 强制内存回收
    gc.collect()
    
    # 分析SHAP重要性，考虑变量对
    print("分析SHAP特征重要性...")
    importance_results = analyze_rf_shap_importance(shap_values, feature_names, model_name, shap_dir)
    gc.collect()
    
    # 如果有年份信息和几何信息，处理年度SHAP输出
    if year_info is not None and not year_info.empty:
        # 生成按年份分组的SHAP输出
        print("生成按年份分组的SHAP输出...")
        try:
            generate_yearly_rf_shap(
                shap_values, X_sample, sample_indices, year_info, 
                feature_names, geo_info, shap_dir
            )
            gc.collect()
        except Exception as e:
            print(f"⚠️ 生成年度SHAP输出出错: {e}")
            traceback.print_exc()
        
        # 生成年度特征重要性
        print("生成年度特征重要性...")
        try:
            generate_yearly_rf_shap_importance(
                shap_values, sample_indices, year_info, feature_names, shap_dir
            )
            gc.collect()
        except Exception as e:
            print(f"⚠️ 生成年度特征重要性出错: {e}")
            traceback.print_exc()
        
        # 生成聚合SHAP输出
        print("生成聚合SHAP输出...")
        try:
            generate_aggregated_rf_shap(
                os.path.join(shap_dir, "yearly_shap"),
                shap_dir,
                feature_names
            )
            gc.collect()
        except Exception as e:
            print(f"⚠️ 生成聚合SHAP输出出错: {e}")
            traceback.print_exc()
        
        # 生成全局SHAP文件
        print("生成全局SHAP文件...")
        try:
            generate_global_rf_shap_file(
                shap_values, sample_indices, year_info, feature_names, geo_info, shap_dir
            )
            gc.collect()
        except Exception as e:
            print(f"⚠️ 生成全局SHAP文件出错: {e}")
            traceback.print_exc()
        
        # 计算气候变量统计
        print("计算气候变量统计信息...")
        try:
            calculate_climate_stats(
                shap_values, sample_indices, year_info, feature_names, geo_info, shap_dir
            )
            gc.collect()
        except Exception as e:
            print(f"⚠️ 计算气候变量统计出错: {e}")
            traceback.print_exc()
    
    print("SHAP分析完成!")
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'importance': importance_results
    }

def create_memmap_features(X_array, filename, mode='w+'):
    """创建内存映射文件用于大规模数据处理"""
    # 创建内存映射文件
    memmap_array = np.memmap(filename, dtype='float32', mode=mode,
                           shape=X_array.shape)
    
    # 写入数据
    if mode in ['w+', 'r+']:
        memmap_array[:] = X_array[:]
        memmap_array.flush()
    
    return memmap_array

def save_metadata_for_shap(data, file_paths, id_year_mapping, geo_info, OUTPUT_DIR):
    """
    保存用于SHAP分析的元数据
    
    参数:
        data: 完整数据集
        file_paths: 数据文件路径列表
        id_year_mapping: ID-年份映射DataFrame
        geo_info: 几何信息DataFrame
        OUTPUT_DIR: 输出目录
    """
    metadata_dir = os.path.join(OUTPUT_DIR, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    print("保存SHAP分析所需的元数据...")
    
    # 1. 保存文件路径列表
    with open(os.path.join(metadata_dir, 'file_paths.json'), 'w') as f:
        # 将路径转换为相对路径存储，便于跨环境使用
        rel_paths = [os.path.basename(p) for p in file_paths]
        json.dump(rel_paths, f)
    
    # 2. 保存ID-年份映射
    if id_year_mapping is not None and not id_year_mapping.empty:
        id_year_mapping.to_pickle(os.path.join(metadata_dir, 'id_year_mapping.pkl'))
    
    # 3. 保存几何信息（如果有）
    if geo_info is not None and not geo_info.empty:
        try:
            # 保存为GeoPackage格式，比Shapefile更能保留更多数据类型
            geo_info.to_file(os.path.join(metadata_dir, 'geo_info.gpkg'), driver='GPKG')
        except Exception as e:
            print(f"保存几何信息为GPKG出错: {e}")
            # 保存为Pickle作为备用
            try:
                geo_info.to_pickle(os.path.join(metadata_dir, 'geo_info.pkl'))
                print("已保存几何信息为PKL格式")
            except Exception as e2:
                print(f"保存几何信息为PKL出错: {e2}")
                # 尝试保存为JSON
                if 'geometry' in geo_info.columns:
                    # 将几何对象转换为WKT字符串
                    geo_info_json = geo_info.copy()
                    geo_info_json['geometry'] = geo_info_json['geometry'].apply(lambda x: str(x) if x else None)
                    geo_info_json.to_json(os.path.join(metadata_dir, 'geo_info.json'))
                    print("已保存几何信息为JSON格式")
    
    # 4. 保存数据集的基本信息
    if data is not None:
        # 保存列信息
        columns_info = {
            'all_columns': list(data.columns),
            'feature_groups': {
                group_name: list(group_info['features']) 
                for group_name, group_info in FEATURE_GROUPS.items()
            },
            'exclude_columns': EXCLUDE_COLUMNS
        }
        with open(os.path.join(metadata_dir, 'columns_info.json'), 'w') as f:
            json.dump(columns_info, f)
    
    # 5. 保存变量对和非对子变量信息
    pairs_info = {
        'all_pairs': ALL_PAIRS,
        'climate_pairs': CLIMATE_PAIRS,
        'non_pair_features': NON_PAIR_FEATURES
    }
    with open(os.path.join(metadata_dir, 'pairs_info.json'), 'w') as f:
        # 由于元组不能直接转为JSON，先转换为列表
        pairs_info_json = {
            'all_pairs': [list(pair) for pair in ALL_PAIRS],
            'climate_pairs': [list(pair) for pair in CLIMATE_PAIRS],
            'non_pair_features': NON_PAIR_FEATURES
        }
        json.dump(pairs_info_json, f)
    
    print(f"元数据已保存到 {metadata_dir}")

def load_metadata_for_shap(OUTPUT_DIR):
    """
    加载SHAP分析所需的元数据
    
    参数:
        OUTPUT_DIR: 输出目录
        
    返回:
        包含元数据的字典
    """
    metadata_dir = os.path.join(OUTPUT_DIR, 'metadata')
    if not os.path.exists(metadata_dir):
        print(f"警告: 元数据目录{metadata_dir}不存在")
        return {}
    
    metadata = {}
    
    # 1. 加载文件路径列表
    file_paths_json = os.path.join(metadata_dir, 'file_paths.json')
    if os.path.exists(file_paths_json):
        with open(file_paths_json, 'r') as f:
            metadata['file_paths'] = json.load(f)
    
    # 2. 加载ID-年份映射
    id_year_mapping_path = os.path.join(metadata_dir, 'id_year_mapping.pkl')
    if os.path.exists(id_year_mapping_path):
        try:
            metadata['id_year_mapping'] = pd.read_pickle(id_year_mapping_path)
            print("✓ 已加载ID-年份映射")
        except Exception as e:
            print(f"加载ID-年份映射出错: {e}")
    
    # 3. 加载几何信息
    geo_info_paths = [
        (os.path.join(metadata_dir, 'geo_info.gpkg'), 'gpkg', lambda p: gpd.read_file(p)),
        (os.path.join(metadata_dir, 'geo_info.pkl'), 'pkl', lambda p: pd.read_pickle(p)),
        (os.path.join(metadata_dir, 'geo_info.json'), 'json', lambda p: pd.read_json(p))
    ]
    
    for path, fmt, load_func in geo_info_paths:
        if os.path.exists(path):
            try:
                metadata['geo_info'] = load_func(path)
                print(f"✓ 已加载几何信息 ({fmt}格式)")
                break
            except Exception as e:
                print(f"加载几何信息 ({fmt}) 出错: {e}")
    
    # 4. 加载列信息
    columns_info_path = os.path.join(metadata_dir, 'columns_info.json')
    if os.path.exists(columns_info_path):
        with open(columns_info_path, 'r') as f:
            metadata['columns_info'] = json.load(f)
            print("✓ 已加载列信息")
    
    # 5. 加载变量对信息
    pairs_info_path = os.path.join(metadata_dir, 'pairs_info.json')
    if os.path.exists(pairs_info_path):
        with open(pairs_info_path, 'r') as f:
            pairs_info_json = json.load(f)
            # 将列表转回元组
            pairs_info = {
                'all_pairs': [tuple(pair) for pair in pairs_info_json['all_pairs']],
                'climate_pairs': [tuple(pair) for pair in pairs_info_json['climate_pairs']],
                'non_pair_features': pairs_info_json['non_pair_features']
            }
            metadata['pairs_info'] = pairs_info
            print("✓ 已加载变量对信息")
    
    return metadata

def load_saved_model(model_dir, model_type='validation_model'):
    """
    加载保存的模型和相关数据
    
    参数:
        model_dir: 模型目录
        model_type: 模型类型 ('validation_model' 或 'all_data_model')
        
    返回:
        模型和相关数据的字典
    """
    models_dir = os.path.join(model_dir, 'models')
    if not os.path.exists(models_dir):
        print(f"警告: 模型目录{models_dir}不存在")
        return None
    
    model_data = {}
    
    # 模型名称从目录名获取
    model_name = os.path.basename(model_dir)
    
    # 1. 加载模型
    model_path = os.path.join(models_dir, f"{model_name}_{model_type}.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data['model'] = pickle.load(f)
            print(f"✓ 已加载{model_type}模型")
        except Exception as e:
            print(f"加载模型{model_path}出错: {e}")
            return None
    else:
        print(f"警告: 模型文件{model_path}不存在")
        return None
    
    # 2. 加载标准化器
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                model_data['scaler'] = pickle.load(f)
            print(f"✓ 已加载标准化器")
        except Exception as e:
            print(f"加载标准化器出错: {e}")
    
    # 3. 加载特征名称
    features_path = os.path.join(models_dir, f"{model_name}_feature_names.pkl")
    if os.path.exists(features_path):
        try:
            with open(features_path, 'rb') as f:
                model_data['feature_names'] = pickle.load(f)
            print(f"✓ 已加载特征名称")
        except Exception as e:
            print(f"加载特征名称出错: {e}")
    
    # 4. 尝试加载测试数据（如果有）
    test_data_path = os.path.join(models_dir, f"{model_name}_test_data.pkl")
    if os.path.exists(test_data_path):
        try:
            with open(test_data_path, 'rb') as f:
                model_data['test_data'] = pickle.load(f)
            print(f"✓ 已加载测试数据")
        except Exception as e:
            print(f"加载测试数据出错: {e}")
    
    return model_data
def save_current_script(OUTPUT_DIR, filename="backup_timesnet_script.py"):
    try:
        # 获取当前脚本的完整路径（仅在运行 .py 文件时有效）
        import inspect
        current_file = inspect.getframeinfo(inspect.currentframe()).filename
        with open(current_file, 'r', encoding='utf-8') as src_file:
            code = src_file.read()

        backup_path = os.path.join(OUTPUT_DIR, filename)
        with open(backup_path, 'w', encoding='utf-8') as backup_file:
            backup_file.write(code)

        print(f"✅ 当前脚本已保存为: {backup_path}")
    except Exception as e:
        print(f"❌ 备份脚本失败: {e}")

def compute_shap_values_in_parallel(model, X_data, batch_size=BATCH_SIZE_SHAP, n_jobs=N_JOBS, 
                                   OUTPUT_DIR=None, model_name="all_vars"):
    """
    计算SHAP值，保留批次文件缓存，使用并行处理避免资源耗尽
    
    参数:
        model: 训练好的随机森林模型
        X_data: 输入特征数据
        batch_size: 每个批次的样本数量
        n_jobs: 并行任务数
        OUTPUT_DIR: 输出目录
        model_name: 模型名称
    """
    import os, time, gc, json, traceback
    import numpy as np
    from joblib import Parallel, delayed
    import shap
    
    # 设置缓存目录
    shap_dir = os.path.join(OUTPUT_DIR, 'shap')
    
    # 创建permanent_cache目录保存批次结果
    cache_dir = os.path.join(shap_dir, "permanent_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"SHAP缓存目录: {cache_dir}")
    print(f"DEBUG - 接收到的shap_dir参数值: {shap_dir}")
    
    # 进度文件路径
    progress_file = os.path.join(cache_dir, f"{model_name}_progress.json")
    
    # 数据基本信息
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1] if len(X_data.shape) > 1 else 1
    n_batches = int(np.ceil(n_samples / batch_size))
    print(f"将 {n_samples} 个样本分成 {n_batches} 个批次，每批 {batch_size} 个样本")
    
    # 加载进度文件，检查是否有已完成的批次
    completed_batches = set()
    batch_params_changed = False
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                saved_batch_size = progress.get('batch_size', batch_size)
                saved_n_samples = progress.get('n_samples', n_samples)
                
                # 检查批次参数是否一致
                if saved_batch_size != batch_size or saved_n_samples != n_samples:
                    print(f"⚠️ 警告: 批次参数已更改")
                    print(f"  原批次大小: {saved_batch_size}, 现批次大小: {batch_size}")
                    print(f"  原样本数: {saved_n_samples}, 现样本数: {n_samples}")
                    batch_params_changed = True
                else:
                    # 如果参数一致，加载已完成的批次
                    completed_batches = set(progress.get('completed_batches', []))
                    print(f"从进度文件恢复: 已完成 {len(completed_batches)}/{n_batches} 个批次")
        except Exception as e:
            print(f"读取进度文件失败，将重新开始: {e}")
    
    # 如果参数变更，警告用户但继续使用之前的缓存
    if batch_params_changed:
        print("参数变更但将尝试继续使用现有缓存文件...")
        # 检查哪些批次已经完成
        completed_batches = set()
        for i in range(n_batches):
            batch_file = os.path.join(cache_dir, f"{model_name}_batch_{i}.npy")
            if os.path.exists(batch_file):
                # 验证文件是否有效
                try:
                    temp = np.load(batch_file, allow_pickle=True)
                    if temp is not None and isinstance(temp, np.ndarray):
                        completed_batches.add(i)
                        print(f"发现有效的批次文件: batch_{i}")
                except:
                    print(f"批次文件 {batch_file} 已损坏，将重新计算")
    
    # 更新进度函数
    def update_progress():
        try:
            with open(progress_file, 'w') as f:
                progress = {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'batch_size': batch_size,
                    'total_batches': n_batches,
                    'completed_batches': sorted(list(completed_batches)),
                    'percent_complete': round(len(completed_batches) / n_batches * 100, 2),
                    'last_update': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"更新进度文件失败: {e}")
    
    # 确定需要处理的批次
    batches_to_process = []
    for i in range(n_batches):
        batch_file = os.path.join(cache_dir, f"{model_name}_batch_{i}.npy")
        if i in completed_batches and os.path.exists(batch_file):
            # 验证文件是否有效
            try:
                temp = np.load(batch_file, allow_pickle=True)
                if temp is not None and isinstance(temp, np.ndarray):
                    continue  # 文件有效，跳过该批次
                else:
                    print(f"批次文件 {batch_file} 内容无效，将重新计算")
                    batches_to_process.append(i)
            except:
                print(f"批次文件 {batch_file} 已损坏，将重新计算")
                batches_to_process.append(i)
        else:
            batches_to_process.append(i)
    
    print(f"需要处理 {len(batches_to_process)}/{n_batches} 个批次")
    
    # 批处理函数
    def compute_batch(batch_idx):
        """计算单个批次的SHAP值"""
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        # 输出和缓存文件路径
        batch_file_base = os.path.join(cache_dir, f"{model_name}_batch_{batch_idx}")
        batch_file = f"{batch_file_base}.npy"
        temp_file = f"{batch_file_base}.tmp"  # 不要在已有.npy的基础上添加.tmp
        
        print(f"处理批次 {batch_idx+1}/{n_batches} (样本 {start_idx}-{end_idx})")
        
        try:
            # 提取批次数据
            if isinstance(X_data, np.memmap):
                batch = np.array(X_data[start_idx:end_idx])
            else:
                batch = X_data[start_idx:end_idx]
            
            # 计算SHAP值
            explainer = shap.TreeExplainer(model)
            batch_values = explainer.shap_values(batch)
            
            # 修复：保存时不让NumPy自动添加扩展名
            with open(temp_file, 'wb') as f:
                np.save(f, batch_values)
            
            # 验证文件
            if not os.path.exists(temp_file):
                print(f"❌ 临时文件创建失败: {temp_file}")
                return False
                
            if os.path.getsize(temp_file) == 0:
                print(f"❌ 临时文件大小为零: {temp_file}")
                os.remove(temp_file)
                return False
            
            # 原子重命名
            if os.path.exists(batch_file):
                os.remove(batch_file)
            os.rename(temp_file, batch_file)
            
            # 清理内存
            del batch, explainer, batch_values
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"❌ 批次 {batch_idx+1}/{n_batches} 处理失败: {e}")
            traceback.print_exc()
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False
    
    # 使用并行处理
    if batches_to_process:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(compute_batch)(i) for i in batches_to_process
        )
        
        # 更新进度
        update_progress()
    else:
        print("✓ 所有批次已完成，无需计算")
    
    # 检查所有批次是否已完成
    all_complete = True
    missing_batches = []
    
    for i in range(n_batches):
        batch_file = os.path.join(cache_dir, f"{model_name}_batch_{i}.npy")
        if not os.path.exists(batch_file):
            all_complete = False
            missing_batches.append(i)
    
    if not all_complete:
        print(f"⚠️ {len(missing_batches)} 个批次未完成。请重新运行程序完成剩余批次。")
        return None
    
    # 所有批次已完成，合并结果
    print("所有批次已完成，合并结果...")
    
    # 使用内存映射文件避免内存溢出
    temp_result_file = os.path.join(cache_dir, f"{model_name}_temp_result.dat")
    if os.path.exists(temp_result_file):
        try:
            os.remove(temp_result_file)
        except:
            pass
    
    result_memmap = np.memmap(temp_result_file, dtype='float32', mode='w+', 
                             shape=(n_samples, n_features))
    
    # 逐个批次合并
    try:
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            # 加载批次
            batch_file = os.path.join(cache_dir, f"{model_name}_batch_{batch_idx}.npy")
            batch_values = np.load(batch_file)
            
            # 复制到结果
            result_memmap[start_idx:end_idx] = batch_values
            
            # 释放内存
            del batch_values
            gc.collect()
        
        # 确保写入完成
        result_memmap.flush()
        
        # 转换为普通数组
        shap_values = np.array(result_memmap[:])
        
        # 释放内存映射
        del result_memmap
        gc.collect()
        
        # 更新进度文件
        with open(progress_file, 'w') as f:
            progress = {
                'n_samples': n_samples,
                'n_features': n_features,
                'batch_size': batch_size,
                'total_batches': n_batches,
                'completed_batches': list(range(n_batches)),
                'final_result': 'completed',
                'last_update': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            json.dump(progress, f, indent=2)
        
        # 删除临时文件（但保留批次文件）
        try:
            os.remove(temp_result_file)
        except:
            pass
        
        print("✓ SHAP值计算和合并完成!")
        return shap_values
        
    except Exception as e:
        print(f"合并批次出错: {e}")
        traceback.print_exc()
        return None


def main():
    # 删除命令行参数处理
    original_output_dir = OUTPUT_DIR  # 确保变量初始化
    # 记录开始时间
    start_time = time.time()
    
    # 创建目录结构
    print("步骤1: 创建目录结构")
    create_directory_structure()
    
    # 获取文件路径
    file_paths = [os.path.join(DATA_DIR, file) for file in FILE_LIST]
    save_current_script(OUTPUT_DIR)
    print(f"✅ 当前脚本已保存为: {OUTPUT_DIR}/backup_timesnet_script.py")
    
    # 如果在指定位置找不到文件，尝试在当前目录查找
    if not all(os.path.exists(path) for path in file_paths):
        csv_files = glob.glob("*.csv")
        if csv_files:
            print(f"未能在指定位置找到文件，将使用当前目录中的{len(csv_files)}个CSV文件。")
            file_paths = csv_files
    
    # 定义关键文件路径
    model_name = 'all_vars'  # 默认使用全部变量模型
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    model_path = os.path.join(model_dir, 'models', f"{model_name}_all_data_model.pkl")
    scaler_path = os.path.join(model_dir, 'models', f"{model_name}_scaler.pkl")
    feature_names_path = os.path.join(model_dir, 'models', f"{model_name}_feature_names.pkl")
    
    # 检查是否存在已训练的模型和必要数据
    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)
    feature_names_exist = os.path.exists(feature_names_path)
    
    # 声明全局变量
    all_results = {}
    data = None
    id_year_mapping = None
    geo_info = None
    
    # 检查关键文件是否存在
    if model_exists and scaler_exists and feature_names_exist:
        print("发现已训练的模型和必要数据，跳过训练阶段")
        
        # 加载模型和相关数据
        model_data = load_saved_model(model_dir, 'all_data_model')
        if model_data is None:
            print("❌ 加载模型失败，将执行完整训练流程")
            model_exists = False
        else:
            print("✅ 成功加载模型和相关数据")
            
            # 加载元数据
            metadata = load_metadata_for_shap(OUTPUT_DIR)
            if metadata:
                if 'id_year_mapping' in metadata:
                    id_year_mapping = metadata['id_year_mapping']
                    print("✅ 已加载ID-年份映射")
                
                if 'geo_info' in metadata:
                    geo_info = metadata['geo_info']
                    print("✅ 已加载地理信息")
            
            # 为SHAP分析准备模型和特征数据
            all_results[model_name] = {
                'models': {'all_data_model': model_data['model']},
                'feature_names': model_data['feature_names']
            }
    
    # 如果没有找到模型或加载失败，执行完整训练流程
    if not model_exists or not scaler_exists or not feature_names_exist or all_results.get(model_name) is None:
        print("需要执行完整训练流程...")
        
        # 加载数据
        print("步骤2: 加载数据")
        data = load_data(file_paths)
        
        if data is None:
            print("没有可用数据。请检查您的文件路径。")
            return
        
        display_progress_info(2, 8, start_time)
        
        # 建立并评估每个特征组的模型
        all_results = {}
        total_groups = len(FEATURE_GROUPS)
        original_output_dir = OUTPUT_DIR
        for i, (group_name, group_info) in enumerate(FEATURE_GROUPS.items(), 1):
            print(f"\n\n{'='*50}")
            print(f"步骤{i+2}/8: 为特征组构建模型: {group_info['name']}")
            print(f"{'='*50}")
            
            # 为每个特征组创建单独的目录路径，不修改全局OUTPUT_DIR
            model_dir = os.path.join(original_output_dir, group_name)
            
            # 准备特征和目标
            X, y, valid_features = prepare_features(data, group_info['features'], target_col='agbd', exclude_columns=EXCLUDE_COLUMNS)
            
            print(f"使用{len(valid_features)}个特征建立该模型")
            
            # 训练和评估模型
            results = train_evaluate_rf_model(X, y, n_folds=5, test_size=0.2)
            
            # 存储结果
            all_results[group_name] = results
            
            # 打印评估指标
            print(f"\n{group_info['name']}的交叉验证平均指标:")
            for metric, value in results['mean_cv_metrics'].items():
                print(f"{metric}: {value:.4f}")
            
            print(f"\n{group_info['name']}的验证指标:")
            for metric, value in results['validation_metrics'].items():
                if metric != 'Feature_Importances':
                    print(f"{metric}: {value:.4f}")
            
            print(f"\n{group_info['name']}的全数据模型指标:")
            for metric, value in results['all_data_metrics'].items():
                if metric != 'Feature_Importances':
                    print(f"{metric}: {value:.4f}")
            
            # 保存指标到文件
            metrics_dir = os.path.join(model_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # 保存折叠指标
            fold_metrics_df = results['fold_metrics'].copy()
            if 'Feature_Importances' in fold_metrics_df.columns:
                fold_metrics_df = fold_metrics_df.drop('Feature_Importances', axis=1)
            fold_metrics_df.to_csv(os.path.join(metrics_dir, f'{group_name}_fold_metrics.csv'), index=False)
            
            # 保存平均CV、验证和全数据指标
            summary_metrics = pd.DataFrame([
                {'模型': '平均CV', **results['mean_cv_metrics']},
                {'模型': '验证集', **{k: v for k, v in results['validation_metrics'].items() if k != 'Feature_Importances'}},
                {'模型': '全数据', **{k: v for k, v in results['all_data_metrics'].items() if k != 'Feature_Importances'}}
            ])
            summary_metrics.to_csv(os.path.join(metrics_dir, f'{group_name}_summary_metrics.csv'), index=False)
            
            # 可视化结果
            visualize_model_results(results, group_name, group_info['name'], model_dir)
            
            # 保存特征重要性分析为TXT
            importance_dir = os.path.join(model_dir, 'importance')
            
            # 保存完整特征重要性分析
            with open(os.path.join(importance_dir, f'{group_name}_full_importance_analysis.txt'), 'w', encoding='utf-8') as f:
                f.write(f"{group_info['name']}模型特征重要性分析\n")
                f.write("="*50 + "\n\n")
                
                f.write("验证模型特征重要性：\n")
                f.write("-"*50 + "\n")
                f.write(results['validation_metrics']['Feature_Importances'].to_string(index=False))
                
                f.write("\n\n全数据模型特征重要性：\n")
                f.write("-"*50 + "\n")
                f.write(results['all_data_metrics']['Feature_Importances'].to_string(index=False))
            
            # 保存模型和相关数据
            save_models(results, group_name, model_dir)
            
            display_progress_info(i+2, 8, start_time)
        
        # 执行模型间配对t检验
        print("步骤6/8: 执行模型间配对t检验")
        ttest_results = perform_paired_ttests(all_results)
        display_progress_info(6, 8, start_time)
        
        # 创建汇总比较
        print("步骤7/8: 创建模型汇总比较")
        summary = summarize_models(all_results)
        display_progress_info(7, 8, start_time)
        
        # 创建README文件
        print("步骤8/8: 创建说明文档")
        create_readme_file()
        display_progress_info(8, 8, start_time)
        
        # 打印最终结果
        print("\n\n=== 模型比较汇总 ===")
        print(summary.to_string())
        print(f"\n结果已保存到'{OUTPUT_DIR}'目录")
    
        # 如果数据中有ID列，创建年份信息映射
        if data is not None and 'ID' in data.columns:
            print("创建年份信息映射...")
            
            # 创建包含ID和年份的DataFrame
            id_data = []
            
            # 从文件名中提取年份信息
            for i, file_path in enumerate(file_paths):
                file_name = os.path.basename(file_path)
                if i < 11:
                    year = 2001 + i  # 2001-2011年
                else:
                    year = 2001 + i + 1  # 跳过2012，从2013开始
                    
                # 读取文件并提取ID
                try:
                    df = pd.read_csv(file_path)
                    for id_val in df['ID'].values:
                        id_data.append({
                            'ID': id_val,
                            'year': year,
                            'ID_year': f"{id_val}_{year}"
                        })
                except Exception as e:
                    print(f"处理文件{file_name}时出错: {e}")
            
            id_year_mapping = pd.DataFrame(id_data)
            
            # 创建几何信息DataFrame
            if 'geometry' in data.columns and 'x' in data.columns and 'y' in data.columns:
                geo_info = data[['ID', 'x', 'y', 'geometry']].copy()
                
                # 转换几何列为shapely对象
                if isinstance(geo_info['geometry'].iloc[0], str):
                    from shapely import wkt
                    geo_info['geometry'] = geo_info['geometry'].apply(
                        lambda x: wkt.loads(x) if isinstance(x, str) else x
                    )
                
                # 转换为GeoDataFrame
                try:
                    geo_info = gpd.GeoDataFrame(geo_info, geometry='geometry')
                except Exception as e:
                    print(f"转换为GeoDataFrame时出错: {e}")
        
        # 保存元数据以便于后续SHAP分析
        save_metadata_for_shap(data, file_paths, id_year_mapping, geo_info, OUTPUT_DIR)
    
    # 打印总运行时间
    total_time = time.time() - start_time
    print(f"\n训练阶段总运行时间: {format_time(total_time)}")
    
    # ===== SHAP分析部分 =====
    # 选择要进行SHAP分析的模型
    target_model_name = model_name  # 默认使用全部变量模型
    
    if target_model_name in all_results:
        print("\n\n" + "="*50)
        print(f"对{target_model_name}模型执行SHAP分析...")
        print("="*50)
        
        # 获取模型
        model = all_results[target_model_name]['models']['all_data_model']
        feature_names = all_results[target_model_name]['feature_names']
        
        # 准备全部数据进行SHAP分析
        X_all = None
        
        # 如果数据还未加载，尝试加载
        if data is None:
            print("尝试从原始文件加载数据...")
            data = load_data(file_paths)
        
        # 使用全部原始数据
        if data is not None:
            print("从原始数据提取特征用于SHAP分析...")
            X_all, _, _ = prepare_features(data, 
                                          FEATURE_GROUPS[target_model_name]['features'], 
                                          target_col='agbd', 
                                          exclude_columns=EXCLUDE_COLUMNS)
            print(f"准备了{len(X_all)}个样本进行SHAP分析")
        else:
            print("⚠️ 无法加载原始数据，SHAP分析无法继续")
            return
        
        # 执行SHAP分析
        try:
            # 使用内存映射临时存储数据，防止内存不足
            X_memmap_path = os.path.join(OUTPUT_DIR, f'{target_model_name}_X_all.memmap')
            X_memmap = create_memmap_features(
                np.array(X_all), X_memmap_path
            )
            
            print(f"创建内存映射文件: {X_memmap_path}")
            
            # 执行SHAP分析
            shap_results = analyze_rf_with_shap(
                model=model,
                X_all_data=X_memmap,
                feature_names=feature_names, 
                OUTPUT_DIR=os.path.join(original_output_dir, target_model_name),
                model_name=target_model_name, 
                sample_size=None,  # 使用全部样本
                year_info=id_year_mapping,
                geo_info=geo_info,
                batch_size=BATCH_SIZE_SHAP,  # 使用全局变量
                n_jobs=10  # 使用所有CPU核心
            )
            
            # 清理内存映射文件
            if os.path.exists(X_memmap_path):
                os.remove(X_memmap_path)
                print(f"已删除临时内存映射文件")
                
            print("SHAP分析完成!")
        except Exception as e:
            print(f"SHAP分析出现错误: {e}")
            traceback.print_exc()
    else:
        print(f"\n⚠️ 未能找到{target_model_name}模型的结果，无法进行SHAP分析")

if __name__ == "__main__":
    main()
