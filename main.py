import os
import gc
import h5py
import datetime
import numpy as np
import pandas as pd
import glob
import geopandas as gpd
from tqdm import tqdm
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, r2_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib
import os.path
from scipy import stats
import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)
import shap
# 启用TF 1.x兼容模式（必须在开头就设置）
import random
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import shapely.wkt
tf.compat.v1.disable_eager_execution()
#from tensorflow import keras
compat = tf.compat.v1
keras = compat.keras
K = keras.backend
Adam = keras.optimizers.Adam
regularizers = keras.regularizers
layers = keras.layers
models = keras.models
callbacks = keras.callbacks
get_session = keras.backend.get_session
from shapely import wkt

#compat.disable_v2_behavior()
EarlyStopping = keras.callbacks.EarlyStopping
tf.compat.v1.keras.backend.get_session

# shap.log.set_level("ERROR")  # 可选：关闭 SHAP 的冗长日志
# 启用TF 1.x兼容模式
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 全局参数设置
INPUT_DIR = r"H:\4.8\8km\两个年份合并计算\自变量+agbd表格"
OUTPUT_DIR = r"H:\6.6\模型\8km——3"
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 40
HIDDEN_DIM = 256  # 新增: 隐藏层维度设置

USE_ORIENT =False  # 是否使用 orient 列作为分类变量，设置为 True 或 False
# 控制是否处理异常值/极端值
USE_IQR_FILTER = False        # 是否使用 IQR 四分位数过滤
USE_WINSORIZE = True          # 是否使用百分位截断（缩尾）
USE_LOG_TRANSFORM = True      # 是否使用 log(x + 1) 转换平滑
USE_ZSCORE_FILTER = True       # 是否使用 Z-score 标准化过滤
L2_REG = 0.0001  # 可调，推荐范围：0.0001 ~ 0.001
SEQ_LEN = 1  # 时间步长（TimesNet 结构）
N_CLUSTERS = 72342  # KMeans 聚类数量（代表性样本选择）
BATCH_SIZE_SHAP = 100000  # 每批多少行
#2全部、3其他、1气候

# 自变量和因变量的字段列表
#气候变量1
FEATURE_COLUMNS1 = ['rx1day','r10mm','cdd','tx10pd','r99totd','sdiid','cddd','rx5d','tn90p','tx90pd','rx1dayd','r99tot',
'r95tot','tn10pd','rx5dd','tn90pd','txnd','txn','tn10p','sdii','tx10p','dtrd','cwd','txxd','tnn','r95totd','r10mmd','r20mm','tnnd',
'tnx','id_1d','cwdd']
#全部变量2
FEATURE_COLUMNS2 = ['rk', 'bts', 'ld', 'nt','dem','water','slope','hli', 'rkd', 'btsd', 'ldd', 'ntd', 'waterd',
'bio6d','rx1day','r10mm','bio19','bio8','bio6','cdd','bio13d',
'bio17d','bio17','bio3d','bio19d','tx10pd','r99totd','sdiid','bio15','cddd','rx5d',
'bio7','tn90p','tx90pd','rx1dayd','r99tot','bio14d','bio9d','r95tot',
'tn10pd','bio14','rx5dd','tn90pd','bio4','bio3','txnd','bio15d','bio13',
'tx90p','r20mmd','txn','tn10p','sdii','tx10p','bio4d','dtrd','bio9','bio2d',
'cwd','txxd','tnn','bio18','bio7d','r95totd','r10mmd','r20mm','bio8d','bio18d',
'tnnd','tnx','id_1d','cwdd']
#其他变量3
FEATURE_COLUMNS = ['rk', 'bts', 'ld', 'nt','dem','water','slope','hli', 'rkd', 'btsd', 'ldd', 'ntd', 'waterd',
'bio6d','bio19','bio8','bio6','bio13d','bio17d','bio17','bio3d','bio19d','bio15',
'bio7','bio14d','bio9d','bio14','bio4','bio3','bio15d','bio13','bio4d','bio9','bio2d','bio18','bio7d','bio8d','bio18d'
]


CLIMATE_PAIRS = [
        ('cdd', 'cddd'), ('cwd', 'cwdd'), ('dtr', 'dtrd'), ('fd', 'fdd'), 
        ('id_1', 'id_1d'), ('prcptot', 'prcptotd'), ('r10mm', 'r10mmd'), 
        ('r20mm', 'r20mmd'), ('r95tot', 'r95totd'), ('r99tot', 'r99totd'), 
        ('rx1day', 'rx1dayd'), ('rx5d', 'rx5dd'), ('sdii', 'sdiid'), 
        ('su', 'sud'), ('tn10p', 'tn10pd'), ('tn90p', 'tn90pd'), 
        ('tnn', 'tnnd'), ('tnx', 'tnxd'), ('tr', 'trd'), 
        ('tx10p', 'tx10pd'), ('tx90p', 'tx90pd'), ('txn', 'txnd'), ('txx', 'txxd')
    ]

# 全部变量对定义

#气候变量1——对子变量
ALL_PAIRS1 = [
    ('cdd', 'cddd'), ('cwd', 'cwdd'), ('dtr', 'dtrd'), ('fd', 'fdd'),
    ('id_1', 'id_1d'), ('prcptot', 'prcptotd'), ('r10mm', 'r10mmd'), ('r20mm', 'r20mmd'),
    ('r95tot', 'r95totd'), ('r99tot', 'r99totd'), ('rx1day', 'rx1dayd'), ('rx5d', 'rx5dd'),
    ('sdii', 'sdiid'), ('su', 'sud'), ('tn10p', 'tn10pd'), ('tn90p', 'tn90pd'),
    ('tnn', 'tnnd'), ('tnx', 'tnxd'), ('tr', 'trd'), ('tx10p', 'tx10pd'), ('tx90p', 'tx90pd'),
    ('txn', 'txnd'), ('txx', 'txxd')    ]

#全部变量2——对子变量
ALL_PAIRS = [
    ('daolu', 'daolud'), ('rk', 'rkd'), ('bts', 'btsd'), ('ld', 'ldd'), ('nt', 'ntd'),
    ('water', 'waterd'), ('cdd', 'cddd'), ('cwd', 'cwdd'), ('dtr', 'dtrd'), ('fd', 'fdd'),
    ('id_1', 'id_1d'), ('prcptot', 'prcptotd'), ('r10mm', 'r10mmd'), ('r20mm', 'r20mmd'),
    ('r95tot', 'r95totd'), ('r99tot', 'r99totd'), ('rx1day', 'rx1dayd'), ('rx5d', 'rx5dd'),
    ('sdii', 'sdiid'), ('su', 'sud'), ('tn10p', 'tn10pd'), ('tn90p', 'tn90pd'),
    ('tnn', 'tnnd'), ('tnx', 'tnxd'), ('tr', 'trd'), ('tx10p', 'tx10pd'), ('tx90p', 'tx90pd'),
    ('txn', 'txnd'), ('txx', 'txxd')
    
    , ('ai3', 'ai3d'),
    ('ai4', 'ai4d'), ('ai5', 'ai5d'), ('ai8', 'ai8d'), ('con3', 'con3d'), ('con4', 'con4d'),
    ('con7', 'con7d'), ('cot1', 'cot1d'), ('cot2', 'cot2d'),
    ('cot3', 'cot3d'), ('cot4', 'cot4d'), ('cot5', 'cot5d'), ('cot7', 'cot7d'), ('cot8', 'cot8d'),
    ('cog', 'cogd'), ('din1', 'din1d'), ('din2', 'din2d'), ('iji1', 'iji1d'), ('iji2', 'iji2d'),
    ('iji3', 'iji3d'), ('iji4', 'iji4d'), ('iji5', 'iji5d'), ('iji7', 'iji7d'), ('iji8', 'iji8d'),
     ('lpi5', 'lpi5d'),('lpi8', 'lpi8d'), ('lsi1', 'lsi1d'), ('lsi2', 'lsi2d'), ('pd1', 'pd1d'), 
    ('pd5', 'pd5d'), ('pd8', 'pd8d'),  ('pladj4', 'pladj4d'),  ('pladj7', 'pladj7d'),
    ('pland7', 'pland7d'),
    ('pr', 'prd'),  ('split1', 'split1d'), ('split2', 'split2d'),
    ('split3', 'split3d'), ('split4', 'split4d'), ('split5', 'split5d'), ('split7', 'split7d'),
    ('split8', 'split8d'), ('ta', 'tad'), ('tca1', 'tca1d'), ('tca2', 'tca2d'), ('tca5', 'tca5d'), ('tca8', 'tca8d')
    
    , ('bio2', 'bio2d'), ('bio3', 'bio3d'), ('bio4', 'bio4d'),  ('bio6', 'bio6d'),
    ('bio7', 'bio7d'), ('bio8', 'bio8d'), ('bio9', 'bio9d'),  ('bio11', 'bio11d'),
    ('bio12', 'bio12d'), ('bio13', 'bio13d'), ('bio14', 'bio14d'), ('bio15', 'bio15d'),
    ('bio16', 'bio16d'), ('bio17', 'bio17d'), ('bio18', 'bio18d'), ('bio19', 'bio19d')
]

#其他变量2——对子变量
ALL_PAIRS3 = [
    ('daolu', 'daolud'), ('rk', 'rkd'), ('bts', 'btsd'), ('ld', 'ldd'), ('nt', 'ntd'),
    ('water', 'waterd'),  ('ai3', 'ai3d'),
    ('ai4', 'ai4d'), ('ai5', 'ai5d'), ('ai8', 'ai8d'), ('con3', 'con3d'), ('con4', 'con4d'),
    ('con7', 'con7d'), ('cot1', 'cot1d'), ('cot2', 'cot2d'),
    ('cot3', 'cot3d'), ('cot4', 'cot4d'), ('cot5', 'cot5d'), ('cot7', 'cot7d'), ('cot8', 'cot8d'),
    ('cog', 'cogd'), ('din1', 'din1d'), ('din2', 'din2d'), ('iji1', 'iji1d'), ('iji2', 'iji2d'),
    ('iji3', 'iji3d'), ('iji4', 'iji4d'), ('iji5', 'iji5d'), ('iji7', 'iji7d'), ('iji8', 'iji8d'),
     ('lpi5', 'lpi5d'),('lpi8', 'lpi8d'), ('lsi1', 'lsi1d'), ('lsi2', 'lsi2d'), ('pd1', 'pd1d'), 
    ('pd5', 'pd5d'), ('pd8', 'pd8d'),  ('pladj4', 'pladj4d'),  ('pladj7', 'pladj7d'),
    ('pland7', 'pland7d'),
    ('pr', 'prd'),  ('split1', 'split1d'), ('split2', 'split2d'),
    ('split3', 'split3d'), ('split4', 'split4d'), ('split5', 'split5d'), ('split7', 'split7d'),
    ('split8', 'split8d'), ('ta', 'tad'), ('tca1', 'tca1d'), ('tca2', 'tca2d'), ('tca5', 'tca5d'), ('tca8', 'tca8d')
    , ('bio2', 'bio2d'), ('bio3', 'bio3d'), ('bio4', 'bio4d'),  ('bio6', 'bio6d'),
    ('bio7', 'bio7d'), ('bio8', 'bio8d'), ('bio9', 'bio9d'),  ('bio11', 'bio11d'),
    ('bio12', 'bio12d'), ('bio13', 'bio13d'), ('bio14', 'bio14d'), ('bio15', 'bio15d'),
    ('bio16', 'bio16d'), ('bio17', 'bio17d'), ('bio18', 'bio18d'), ('bio19', 'bio19d')
]


# 非对子变量
NON_PAIR_FEATURES = ['dem', 'slope', 'hli']
# 目标变量         
TARGET_COLUMN = ['agbd']
#相关计算后舍弃的变量
TARGET_COLUMN2 = [
    'ai1', 'ai1d', 'ai2', 'ai2d', 'ai7', 'ai7d',
    'ca1', 'ca1d', 'ca2', 'ca2d', 'ca3', 'ca3d', 'ca4', 'ca4d', 'ca5', 'ca5d', 'ca7', 'ca7d', 'ca8', 'ca8d',
    'con1', 'con1d', 'con2', 'con2d', 'con5', 'con5d', 'con8', 'con8d',
    'din3', 'din3d', 'din4', 'din4d', 'din5', 'din5d', 'din7', 'din7d', 'din8', 'din8d',
    'lpi1', 'lpi1d', 'lpi2', 'lpi2d', 'lpi3', 'lpi3d', 'lpi4', 'lpi4d', 'lpi7', 'lpi7d',
    'lsi3', 'lsi3d', 'lsi4', 'lsi4d', 'lsi5', 'lsi5d', 'lsi7', 'lsi7d', 'lsi8', 'lsi8d',
    'np1', 'np1d', 'np2', 'np2d', 'np3', 'np3d', 'np4', 'np4d', 'np5', 'np5d', 'np7', 'np7d', 'np8', 'np8d',
    'pd2', 'pd2d', 'pd3', 'pd3d', 'pd4', 'pd4d', 'pd7', 'pd7d',
    'pladj1', 'pladj1d', 'pladj2', 'pladj2d', 'pladj3', 'pladj3d', 'pladj5', 'pladj5d', 'pladj8', 'pladj8d',
    'pland1', 'pland1d', 'pland2', 'pland2d', 'pland3', 'pland3d', 'pland4', 'pland4d', 'pland5', 'pland5d', 'pland8', 'pland8d',
    'shdi', 'shdid', 'sidi', 'sidid',
    'tca3', 'tca3d', 'tca4', 'tca4d', 'tca7', 'tca7d',
    'bio1', 'bio1d', 'bio5', 'bio5d', 'bio10', 'bio10d'
]
# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "attributions"), exist_ok=True)
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

# KGE评估指标
def kge(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    cov = K.mean((y_true - mean_true) * (y_pred - mean_pred))
    r = cov / (std_true * std_pred + K.epsilon())
    beta = mean_pred / (mean_true + K.epsilon())
    alpha = std_pred / (std_true + K.epsilon())
    return 1.0 - K.sqrt(K.square(r - 1.0) + K.square(alpha - 1.0) + K.square(beta - 1.0))
def remove_outliers_zscore(df, cols, threshold=5.0):
    for col in cols:
        if col in df.columns:
            z_scores = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            df = df[(z_scores.abs() <= threshold)]
    return df
def clip_outliers_zscore(df, cols, threshold=5.0):
    for col in cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std() + 1e-8  # 避免除以0
            lower = mean - threshold * std
            upper = mean + threshold * std
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df

def categorize_orient(angle):
    try:
        angle = float(angle)
        if angle < 0:
            return 'flat'
        elif angle <= 22.5 or angle > 337.5:
            return 'N'
        elif angle <= 67.5:
            return 'NE'
        elif angle <= 112.5:
            return 'E'
        elif angle <= 157.5:
            return 'SE'
        elif angle <= 202.5:
            return 'S'
        elif angle <= 247.5:
            return 'SW'
        elif angle <= 292.5:
            return 'W'
        else:
            return 'NW'
    except:
        return 'unknown'
def remove_outliers_iqr(df, cols):
    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df
# ==================== 添加：分布分析函数 ====================
def analyze_feature_distributions(df, feature_columns, output_dir):
    import matplotlib.pyplot as plt
    from scipy.stats import iqr

    dist_dir = os.path.join(output_dir, "figures", "feature_distributions")
    os.makedirs(dist_dir, exist_ok=True)

    for col in feature_columns:
        if col not in df.columns:
            continue

        try:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if values.empty:
                continue

            # IQR 异常值区间
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr_val = q3 - q1
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            outlier_mask = (values < lower_bound) | (values > upper_bound)
            outlier_ratio = outlier_mask.mean() * 100

            # 画图
            plt.figure(figsize=(8, 4))
            sns.histplot(values, bins=50, kde=True, color='skyblue', edgecolor='black')
            plt.axvline(lower_bound, color='red', linestyle='--', label='IQR 下界')
            plt.axvline(upper_bound, color='red', linestyle='--', label='IQR 上界')
            plt.title(f"{col} 分布（异常值比例: {outlier_ratio:.1f}%）")
            plt.xlabel(col)
            plt.ylabel("频数")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f"{col}.png"))
            plt.close()

        except Exception as e:
            print(f"❌ 分析失败: {col}，原因: {e}")
def winsorize(df, cols, lower=0.01, upper=0.99):
    for col in cols:
        if col in df.columns:
            lower_q = df[col].quantile(lower)
            upper_q = df[col].quantile(upper)
            df[col] = np.clip(df[col], lower_q, upper_q)
    return df

def log_transform(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df
# Huber损失函数
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

# 数据加载和预处理
def load_data_files():
    file_names = [
        "热点01_00.csv", "热点02_01.csv", "热点03_02.csv", "热点04_03.csv", 
        "热点05_04.csv", "热点06_05.csv", "热点07_06.csv", "热点08_07.csv", 
        "热点09_08.csv", "热点10_09.csv", "热点11_10.csv", "热点13_11.csv", 
        "热点14_13.csv", "热点15_14.csv", "热点16_15.csv", "热点17_16.csv", 
        "热点18_17.csv", "热点19_18.csv"
    ]
    
    all_files = []
    for file_name in file_names:
        file_path = os.path.join(INPUT_DIR, file_name)
        if os.path.exists(file_path):
            all_files.append(file_path)
    return all_files

def preprocess_data(all_files):
    """
    预处理数据函数 - 加载CSV文件，处理特征，标准化，并准备TimesNet格式数据
    
    参数:
        all_files: 所有CSV数据文件的路径列表
        
    返回:
        X_scaled: 标准化并调整为3D格式的特征数据 (样本数, 时间步, 特征数)
        y_scaled: 标准化的目标变量
        pixel_ids: 像素/样本ID
        geo_info: 地理信息字典
        feature_columns: 最终使用的特征列名称
        scaler_X: X标准化器
        scaler_y: y标准化器
        gdf_full: 包含几何信息的GeoDataFrame
        id_year_mapping: ID-年份映射DataFrame
    """
    all_data = []
    file_years = []  # 用于记录文件名和年份映射
    # 定义需要排除的列和目标列
    exclude_columns = ['ID', 'geometry', 'x', 'y']
    target_column = TARGET_COLUMN[0]
    
    for i, file_path in tqdm(enumerate(all_files), desc="加载数据集"):
        df = pd.read_csv(file_path)
        
        # === 添加年份标识 ===
        # === 添加年份标识 - 修复版 ===
        if i < 11:
            year = 2001 + i  # 2001-2011年
        elif i >= 11:
            year = 2001 + i + 1  # 跳过2012年，从2013年开始
        df['year'] = year
        df['ID_year'] = df['ID'].astype(str) + "_" + str(year)
        file_name = os.path.basename(file_path)
        file_years.append((file_name, year))
        # 定义元数据列
        metadata_cols = ['year', 'ID_year']
        
        # 选择需要的列，明确分离特征和元数据
        selected_cols = FEATURE_COLUMNS + TARGET_COLUMN + exclude_columns + metadata_cols
        df = df[[col for col in selected_cols if col in df.columns]]
        
        # 处理 orient：分类 + One-Hot 编码（如果 USE_ORIENT 为 True）
        if USE_ORIENT and 'orient' in df.columns:
            try:
                orient_series = pd.to_numeric(df['orient'], errors='coerce')
                df['orient'] = orient_series
                df['orient_cat'] = orient_series.apply(categorize_orient)
                df = pd.get_dummies(df, columns=['orient_cat'])
                df.drop(columns=['orient'], inplace=True)
            except Exception as e:
                print(f"⚠️ orient列处理失败，文件: {file_path}，错误：{e}")
        else:
            # 如果不使用 orient 列，直接去除它
            df.drop(columns=['orient'], errors='ignore', inplace=True)
        
        # 所有非 ID/geometry 列转为数值
        for col in df.columns:
            if col not in ['ID', 'geometry']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 填充缺失值
        df.fillna(0, inplace=True)
        
        # ==== 应用异常值处理方法（可控开关） ====
        if USE_IQR_FILTER:
            df = remove_outliers_iqr(df, FEATURE_COLUMNS)
        if USE_WINSORIZE:
            df = winsorize(df, FEATURE_COLUMNS)
        if USE_LOG_TRANSFORM:
            df = log_transform(df, FEATURE_COLUMNS)
        if USE_ZSCORE_FILTER:
            df = clip_outliers_zscore(df, FEATURE_COLUMNS)
        
        all_data.append(df)
    print("\n===== 文件与年份映射 =====")
    for file_name, year in file_years:
        print(f"{file_name} → {year}年")
    # 合并所有年份
    full_data = pd.concat(all_data, ignore_index=True)
    
    # 保存ID-年份映射
    id_year_mapping = full_data[['ID', 'year', 'ID_year']].copy()
    
    # ===== 关键改进1: 明确排除所有非特征列 =====
    # 定义需要排除的所有列
    exclude_all = ['ID', 'geometry', 'x', 'y', 'year', 'ID_year', 'dg', 'dgd', 'wsdi', 'wsdid','orient'] + TARGET_COLUMN + TARGET_COLUMN2
    
    # 创建特征数据框 - 只包含FEATURE_COLUMNS中存在的列
    feature_cols_in_data = [col for col in FEATURE_COLUMNS if col in full_data.columns]
    feature_data = full_data[feature_cols_in_data].copy()
    
    # === 打印详细调试信息 ===
    print(f"\n===== 特征处理详情 =====")
    print(f"原始数据列数量: {len(full_data.columns)}")
    print(f"FEATURE_COLUMNS定义列数量: {len(FEATURE_COLUMNS)}")
    print(f"实际可用特征列数量: {len(feature_cols_in_data)}")
     # === 打印调试信息 ===
    print("原始数据列: ", list(full_data.columns))
    print("用于特征的列: ", list(feature_data.columns))
    # 检查特征列差异
    missing_features = set(FEATURE_COLUMNS) - set(feature_cols_in_data)
    if missing_features:
        print(f"警告: 在数据中缺少 {len(missing_features)} 个定义的特征列:")
        print(f"  {sorted(list(missing_features))}")
    
    # 检查数据中的非预期列
    extra_cols = [col for col in full_data.columns 
                 if col not in FEATURE_COLUMNS 
                 and col not in exclude_all]
    if extra_cols:
        print(f"发现 {len(extra_cols)} 个非预期列 (不在FEATURE_COLUMNS中):")
        print(f"  {sorted(extra_cols)}")
    
    # ===== 准备地理信息 =====
    # 提取地理信息
    gdf_full = full_data[['ID', 'geometry', 'x', 'y']].copy()
    
    # 将 WKT 字符串转换为 shapely 几何，并升级为 GeoDataFrame
    from shapely import wkt
    gdf_full['geometry'] = gdf_full['geometry'].apply(
        lambda w: wkt.loads(w) if isinstance(w, str) and w.strip() else w
    )
    gdf_full = gpd.GeoDataFrame(gdf_full, geometry='geometry', crs='EPSG:3857')
    
    # 提取用于SHAP分析的相关信息
    pixel_ids = full_data['ID'].values
    geo_info = full_data[['ID', 'x', 'y', 'geometry']].to_dict(orient='records')
    
    # ===== 准备目标变量 =====
    y = full_data[target_column].values
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # ===== 关键改进2: 正确处理2D到3D的转换 =====
    # 1. 创建2D特征矩阵，只使用FEATURE_COLUMNS中存在的列
    X_2d = feature_data.values
    
    # 2. 标准化2D数据
    scaler_X = StandardScaler()
    X_scaled_2d = scaler_X.fit_transform(X_2d)
    
    # 3. 重塑为3D格式 (samples, sequence_length, features)
    X_scaled = np.zeros((len(feature_data), SEQ_LEN, len(FEATURE_COLUMNS)))
    
    # 填充特征值到正确的位置
    for i, feature in enumerate(feature_cols_in_data):
        feature_idx = FEATURE_COLUMNS.index(feature)
        X_scaled[:, 0, feature_idx] = X_scaled_2d[:, i]
    
    # 最终确认形状
    print(f"\n===== 最终数据形状 =====")
    print(f"X_scaled: {X_scaled.shape} (样本数, 时间步, 特征数)")
    print(f"y_scaled: {y_scaled.shape} (样本数)")
    
    # 返回最终结果 - 使用FEATURE_COLUMNS作为特征列
    return X_scaled, y_scaled, pixel_ids, geo_info, FEATURE_COLUMNS, scaler_X, scaler_y, gdf_full, id_year_mapping# 创建TimesNet模型
def build_simplified_timesnet_model(feature_dim, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, num_blocks=3, learning_rate=1e-4):
    """
    构建简化版TimesNet模型，避免保存问题
    """
    # 创建输入层
    inputs = keras.layers.Input(shape=(seq_len, feature_dim))
    
    # 初始特征转换 - 添加L2正则化
    x = keras.layers.Dense(hidden_dim, activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = keras.layers.Dropout(0.3)(x)  # 增加dropout比例
    
    # 简化版多尺度特征提取
    for i in range(num_blocks):
        # 各卷积层添加L2正则化
        conv3 = keras.layers.Conv1D(hidden_dim, kernel_size=3, padding='same', activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001))(x)
        conv5 = keras.layers.Conv1D(hidden_dim, kernel_size=5, padding='same', activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001))(x)
        conv7 = keras.layers.Conv1D(hidden_dim, kernel_size=7, padding='same', activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001))(x)
        
        # 合并和降维步骤保持不变
        concat = keras.layers.Concatenate()([conv3, conv5, conv7])
        reduced = keras.layers.Conv1D(hidden_dim, kernel_size=1)(concat)
        reduced = keras.layers.BatchNormalization()(reduced)
        reduced = keras.layers.Activation('relu')(reduced)
        
        # 残差连接后添加dropout
        x = keras.layers.Add()([x, reduced])
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)  # 每个块后添加dropout
    
    # 全局特征提取后的层也增加正则化
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(hidden_dim//2, activation='relu', 
                         kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)  # 增加最终dropout比例
    
    # 输出层
    outputs = keras.layers.Dense(1)(x)
    
    # 创建模型
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    
    # 编译模型
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5,clipnorm=1.0)#clipnorm=1.0-5.0,clipvalue=0.5-5.0
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse', kge])
    
    return model

def get_callbacks(output_dir, scheduler_type='plateau'):
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, "models", "best_timesnet_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Plateau 学习率调度
    lr_scheduler_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.9,
        patience=5,
        min_lr=1e-9,
        verbose=1
    )

    # Cosine Decay Restart 学习率调度（TF 1.x 兼容）
    cosine_lr = keras.experimental.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=100,
        t_mul=1.2,
        m_mul=0.7,
        alpha=1e-8
    )
    lr_scheduler_cosine = keras.callbacks.LearningRateScheduler(
        lambda epoch: float(cosine_lr(epoch))
    )

    if scheduler_type == 'plateau':
        return [early_stop, checkpoint, lr_scheduler_plateau]
    elif scheduler_type == 'cosine':
        return [early_stop, checkpoint, lr_scheduler_cosine]
    else:
        return [early_stop, checkpoint]

def train_and_evaluate_model(X_scaled, y_scaled, feature_columns):
    step_ahead = 1
    window_size = 1
    num_samples = X_scaled.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_size = int(0.8 * num_samples)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    X_train, y_train = X_scaled[train_indices], y_scaled[train_indices]
    X_test, y_test = X_scaled[test_indices], y_scaled[test_indices]

    X_train_windows, y_train_targets, X_test_windows, y_test_targets = [], [], [], []

    # 替代旧的滑窗构造逻辑
    X_train_windows = X_train
    y_train_targets = y_train
    X_test_windows = X_test
    y_test_targets = y_test




    X_train_windows, y_train_targets = np.array(X_train_windows), np.array(y_train_targets)
    X_test_windows, y_test_targets = np.array(X_test_windows), np.array(y_test_targets)

    # 定义余弦退火学习率策略
    learning_rate = 1e-4  # 使用固定学习率，因为在TF 1.x中使用cosine decay可能会有兼容性问题
    
    # 使用固定学习率创建模型
    model = build_simplified_timesnet_model(
    feature_dim=X_scaled.shape[2],
    seq_len=SEQ_LEN,
    hidden_dim=HIDDEN_DIM,
    learning_rate=1e-4
)
    #model = create_timesnet_model((1, X_scaled.shape[2]), step_ahead=step_ahead, window_size=window_size, hidden_dim=HIDDEN_DIM, learning_rate=learning_rate)
    # 设置回调函数
    # 选择学习率调度器：'plateau' 或 'cosine'
    callbacks_list = get_callbacks(OUTPUT_DIR, scheduler_type='plateau')

    # 模型训练
    history = model.fit(
        X_train_windows, y_train_targets, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_split=0.2, 
        callbacks=callbacks_list,
        verbose=1
    )


    # 保存最终模型
    model.save(os.path.join(OUTPUT_DIR, "models", "timesnet_final_model.h5"))

    # 绘制训练过程中的Loss、MSE和MAE图表
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('Huber损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['mse'], label='训练MSE')
    plt.plot(history.history['val_mse'], label='验证MSE')
    plt.title('均方误差(MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['mae'], label='训练MAE')
    plt.plot(history.history['val_mae'], label='验证MAE')
    plt.title('平均绝对误差(MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "training_history.png"))
    plt.close()
    # 在train_and_evaluate_model函数的最后
    with open(os.path.join(OUTPUT_DIR, "models", "training_features.json"), "w") as f:
        json.dump({"feature_columns": feature_columns}, f)
    return model, X_test_windows, y_test_targets, history

# 训练模型
def evaluate_model_performance(model, X_test_windows, y_test_targets, scaler_y, output_dir):
    from scipy.stats import ttest_1samp
    y_pred_scaled = model.predict(X_test_windows)
    y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_reshaped).flatten()
    y_true = scaler_y.inverse_transform(y_test_targets.reshape(-1, 1)).flatten()

    # 基础指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 其他指标
    rrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else np.nan
    bias = np.mean(y_pred - y_true)
    pbias = np.sum(y_pred - y_true) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else np.nan
    nse = 1 - (np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    def kge_score(obs, sim):
        r = np.corrcoef(obs, sim)[0, 1]
        std_obs, std_sim = np.std(obs), np.std(sim)
        mean_obs, mean_sim = np.mean(obs), np.mean(sim)
        alpha = std_sim / (std_obs + 1e-10)
        beta = mean_sim / (mean_obs + 1e-10)
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    kge = kge_score(y_true, y_pred)
    t_stat, p_value = ttest_1samp(y_pred - y_true, 0)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'KGE': kge,
        'rRMSE': rrmse,
        'BIAS': bias,
        'PBIAS': pbias,
        'NSE': nse,
        't-test p-value': p_value
    }

    # 输出
    print("\n===== Model Performance =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    with open(os.path.join(output_dir, "models", "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k} = {v:.4f}\n")

    return y_pred, metrics
def plot_training_history(history, output_dir):
    history_dict = history.history
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history_dict['loss'], label='Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if 'mae' in history_dict:
        plt.subplot(1, 3, 2)
        plt.plot(history_dict['mae'], label='MAE')
        plt.plot(history_dict['val_mae'], label='Val MAE')
        plt.title('MAE vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

    if 'mse' in history_dict:
        plt.subplot(1, 3, 3)
        plt.plot(history_dict['mse'], label='MSE')
        plt.plot(history_dict['val_mse'], label='Val MSE')
        plt.title('MSE vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'model_training_plots.png'))
    plt.close()
# 使用 GradientExplainer 进行 SHAP 归因分析
def select_representative_samples(X, n_clusters=N_CLUSTERS):
    """
    使用KMeans聚类选择代表性样本
    
    Args:
        X: 训练数据，形状为 [n_samples, seq_len, feature_dim]
        n_clusters: 聚类数量，也是要选择的样本数量
    
    Returns:
        代表性样本的数组，形状与输入相同
    """
    # 获取原始形状信息
    n_samples, seq_len, feature_dim = X.shape
    from sklearn.cluster import MiniBatchKMeans
    # 如果样本数小于聚类数，直接返回所有样本
    if n_samples <= n_clusters:
        print(f"样本数({n_samples})小于等于请求的聚类数({n_clusters})，返回所有样本")
        return X
        
    # 将3D数据重塑为2D，以便进行聚类
    X_reshaped = X.reshape(n_samples, -1)
    
    print(f"开始KMeans聚类 (k={n_clusters})...")
    kmeans = MiniBatchKMeans(
    n_clusters=min(n_clusters, n_samples),
    batch_size=4096,  # 调整批量大小以适应内存
    random_state=27,
    verbose=1
)
    
    # 使用进度条显示聚类进度
    from tqdm import tqdm
    pbar = tqdm(total=100, desc="KMeans聚类")
    
    # 自定义回调函数来更新进度条
    def kmeans_callback(progress):
        pbar.update(int(progress * 100) - pbar.n)
    
    # 执行聚类
    try:
        # 如果KMeans支持verbose_callback参数（较新版本的scikit-learn）
        kmeans.fit(X_reshaped, verbose_callback=kmeans_callback)
    except TypeError:
        # 否则直接执行，不显示进度
        pbar.update(10)  # 显示启动进度
        kmeans.fit(X_reshaped)
        pbar.update(90)  # 完成
    
    pbar.close()
    
    # 找到最接近每个聚类中心的样本点
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_reshaped)
    
    # 返回原始形状的代表性样本
    representative_samples = X[closest]
    
    print(f"已选择 {len(representative_samples)} 个代表性样本")
    return representative_samples
# 创建SHAP解释器
def create_shap_explainer(model, X_train_s, sess, use_kmeans=True, n_clusters=N_CLUSTERS):
    """创建SHAP解释器，使用KMeans选择背景样本"""
    
    # 选择背景数据
    if use_kmeans:
        print("使用KMeans聚类选择代表性背景样本...")
        background_data = select_representative_samples(X_train_s, n_clusters=n_clusters)
    else:
        # 原始的随机选择方法
        background_size = min(N_CLUSTERS, X_train_s.shape[0])
        idx_bg = np.random.choice(X_train_s.shape[0], background_size, replace=False)
        background_data = X_train_s[idx_bg]
        print(f"随机选择了 {len(background_data)} 个背景样本")
    
    # 创建TF 1.x兼容的输入占位符
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, X_train_s.shape[1], X_train_s.shape[2]])
    
    # 创建GradientExplainer
    print("创建SHAP GradientExplainer...")
    try:
        # 获取模型的输入张量（原始模型的第一个输入）
        model_inputs = model.inputs[0]
        # 获取模型的输出张量（原始模型的第一个输出）
        model_outputs = model.outputs[0]
        
        # 尝试直接使用模型的输入输出张量
        explainer = shap.GradientExplainer(
            (model_inputs, model_outputs),
            background_data,
            session=sess
        )
        return explainer, None  # 返回解释器和None（因为没有使用占位符）
    except Exception as e:
        print(f"直接使用模型张量失败: {e}")
        print("尝试使用占位符...")
        
        # 创建一个计算图，复用模型的权重
        with tf.compat.v1.variable_scope('shap_model', reuse=tf.compat.v1.AUTO_REUSE):
            # 使用占位符作为输入
            lstm_out = keras.layers.LSTM(64)(input_ph)
            output_tensor = keras.layers.Dense(1)(lstm_out)
            
            # 尝试使用占位符和输出张量
            explainer = shap.GradientExplainer(
                (input_ph, output_tensor),
                background_data,
                session=sess
            )
        return explainer, input_ph  # 返回解释器和占位符
# 计算SHAP值
def calculate_all_shap_values(explainer, X_all, input_ph=None):
    """分批计算所有样本的SHAP值，支持占位符"""
    batch_size = BATCH_SIZE_SHAP
    shap_values_list = []
    total_samples = X_all.shape[0]
    success_count = 0
    
    for start_idx in tqdm(range(0, total_samples, batch_size), desc="计算SHAP值"):
        end_idx = min(start_idx + batch_size, total_samples)
        X_batch = X_all[start_idx:end_idx]
        
        try:
            # 根据是否使用占位符选择不同的调用方式
            if input_ph is not None:
                # 使用占位符时的调用方式
                shap_batch_values = explainer.shap_values(X_batch, feed_dict={input_ph: X_batch})
            else:
                # 直接调用方式
                shap_batch_values = explainer.shap_values(X_batch)
            
            # 处理返回值
            if isinstance(shap_batch_values, list):
                shap_batch = shap_batch_values[0]
            else:
                shap_batch = shap_batch_values
                
            shap_values_list.append(shap_batch)
            success_count += 1
            print(f"✓ 批次 {start_idx}-{end_idx} 计算成功，形状: {shap_batch.shape}")
            
        except Exception as e:
            print(f"批次 {start_idx}-{end_idx} 计算失败: {e}")
        
        # 清理内存
        gc.collect()
    
    # 合并所有批次的SHAP值
    if not shap_values_list:
        raise ValueError("所有批次SHAP计算均失败!")
        
    shap_values_all = np.concatenate(shap_values_list, axis=0)
    print(f"SHAP计算完成! 成功处理 {success_count} 个批次，总形状: {shap_values_all.shape}")
    
    return shap_values_all
# 生成年度SHAP输出
def generate_yearly_shap_outputs(shap_values, X_all, id_year_mapping, feature_columns_adjusted, gdf_full, output_dir):
    """生成年度SHAP输出 - 修改版，生成两类输出"""
    # 确保输出目录存在
    yearly_output_dir = os.path.join(output_dir, "yearly_shap")
    os.makedirs(yearly_output_dir, exist_ok=True)
    
    # 将3D SHAP值转换为2D
    if len(shap_values.shape) == 3:
        shap_combined = np.sum(shap_values, axis=1)
    else:
        seq_len = X_all.shape[1] 
        feature_dim = X_all.shape[2]
        shap_combined = shap_values.reshape(-1, seq_len, feature_dim)
        shap_combined = np.sum(shap_combined, axis=1)
    
    # 按年份处理
    unique_years = id_year_mapping['year'].unique()
    
    for year in tqdm(unique_years, desc="生成年度SHAP文件"):
        # 创建年份目录
        year_dir = os.path.join(yearly_output_dir, f"year_{year}")
        os.makedirs(year_dir, exist_ok=True)
        
        year_mask = id_year_mapping['year'] == year
        year_indices = np.where(year_mask)[0]
        
        if len(year_indices) == 0:
            print(f"警告: {year}年没有样本")
            continue
        
        # 获取该年份的SHAP值
        year_shap = shap_combined[year_indices]
        
        # ========== 类别1: 原始SHAP值 ==========
        # 基础列
        orig_data_dict = {
            'ID': id_year_mapping.loc[year_indices, 'ID'].values,
            'year': year,
            'geometry': gdf_full.loc[id_year_mapping.loc[year_indices].index, 'geometry'].values
        }
        
        # 添加所有SHAP原始列
        for i, col_name in enumerate(feature_columns_adjusted):
            if i < year_shap.shape[1]:
                orig_data_dict[col_name] = year_shap[:, i]
        
        # 创建原始值GeoDataFrame
        orig_gdf = gpd.GeoDataFrame(orig_data_dict, geometry='geometry')
        
        # 计算total列 (所有特征的综合影响)
        # 对非对子变量计算绝对值，仅使用存在的非对子变量
        non_pair_abs_sum = np.zeros(len(year_indices))
        valid_non_pairs = get_valid_features(NON_PAIR_FEATURES, orig_gdf.columns)
        if valid_non_pairs:
            for col in valid_non_pairs:
                non_pair_abs_sum += np.abs(orig_gdf[col].values)
        else:
            print("警告: 计算total时未找到任何非对子变量")
        
        # 对所有对子变量计算绝对值均值
        pair_abs_mean_sum = np.zeros(len(year_indices))
        for base, derived in ALL_PAIRS:
            if base in orig_gdf.columns and derived in orig_gdf.columns:
                pair_abs = (np.abs(orig_gdf[base].values) + np.abs(orig_gdf[derived].values)) / 2
                pair_abs_mean_sum += pair_abs
        
        # 添加total列
        orig_gdf['total'] = non_pair_abs_sum + pair_abs_mean_sum
        
        # ========== 类别2: 统计值 ==========
        # 基础列
        stats_data_dict = {
            'ID': id_year_mapping.loc[year_indices, 'ID'].values,
            'year': year,
            'geometry': gdf_full.loc[id_year_mapping.loc[year_indices].index, 'geometry'].values
        }
        
        # 计算所有对子的ma和ms值
        ma_values = {}
        ms_values = {}
        
        for base, derived in ALL_PAIRS:
            if base in orig_gdf.columns and derived in orig_gdf.columns:
                # 方向值 (ma)
                ma_values[f"{base}ma"] = (orig_gdf[base] + orig_gdf[derived]) / 2
                
                # 强度值 (ms)
                ms_values[f"{base}ms"] = (np.abs(orig_gdf[base]) + np.abs(orig_gdf[derived])) / 2
        
        # 添加所有ma和ms列
        stats_data_dict.update(ma_values)
        stats_data_dict.update(ms_values)
        
        # 创建统计值GeoDataFrame
        stats_gdf = gpd.GeoDataFrame(stats_data_dict, geometry='geometry')
        
        # 计算totalma列 (所有ma值与非对子变量的原始值之和)
        non_pair_sum = np.zeros(len(year_indices))
        if valid_non_pairs:  # 重用前面获取的列表
            for col in valid_non_pairs:
                non_pair_sum += orig_gdf[col].values
        else:
            print("警告: 计算totalma时未找到任何非对子变量")
        
        ma_sum = np.zeros(len(year_indices))
        for key in ma_values.keys():
            ma_sum += stats_gdf[key].values
        
        # 添加totalma列
        stats_gdf['totalma'] = non_pair_sum + ma_sum
        
        # ========== 保存文件 ==========
        # 1. 保存原始值CSV文件
        orig_csv_path = os.path.join(year_dir, f"shap_original_{year}.csv")
        orig_gdf.to_csv(orig_csv_path, index=False)
        print(f"✓ {year}年原始SHAP保存为CSV: {orig_csv_path}")
        
        # 2. 保存统计值CSV文件
        stats_csv_path = os.path.join(year_dir, f"shap_stats_{year}.csv")
        stats_gdf.to_csv(stats_csv_path, index=False)
        print(f"✓ {year}年统计SHAP保存为CSV: {stats_csv_path}")
        
        # 3. 分割保存原始值SHP文件
        orig_shp_base = os.path.join(year_dir, f"shap_original_{year}")
        split_and_save_shapefile(orig_gdf, orig_shp_base, max_fields=250)
        
        # 4. 分割保存统计值SHP文件
        stats_shp_base = os.path.join(year_dir, f"shap_stats_{year}")
        split_and_save_shapefile(stats_gdf, stats_shp_base, max_fields=250)
def get_valid_features(feature_list, all_features):
    """返回在all_features中实际存在的特征子集"""
    return [f for f in feature_list if f in all_features]

def generate_aggregated_shap_outputs(yearly_shaps_dir, output_dir, feature_columns_adjusted):
    """基于年度SHAP生成聚合SHAP文件 - 修改版"""
    print("生成聚合SHAP输出...")
    agg_dir = os.path.join(output_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    
    # 1. 获取所有年份文件夹
    year_dirs = [os.path.join(yearly_shaps_dir, d) for d in os.listdir(yearly_shaps_dir) 
                if os.path.isdir(os.path.join(yearly_shaps_dir, d)) and d.startswith('year_')]
    
    if not year_dirs:
        print("⚠️ 未找到年度SHAP文件夹，无法生成聚合结果")
        return
    
    # 2. 为每个年份加载原始值和统计值数据
    all_years_original = []  # 存储原始值数据
    all_years_stats = []     # 存储统计值数据
    
    for year_dir in tqdm(year_dirs, desc="加载年度SHAP数据"):
        year = os.path.basename(year_dir).replace('year_', '')
        
        # 加载原始值CSV
        orig_csv = os.path.join(year_dir, f"shap_original_{year}.csv")
        if os.path.exists(orig_csv):
            try:
                orig_df = pd.read_csv(orig_csv)
                # 将geometry列转换为WKT字符串表示
                if 'geometry' in orig_df.columns:
                    orig_df['geometry'] = orig_df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
                    orig_df = gpd.GeoDataFrame(orig_df, geometry='geometry')
                all_years_original.append(orig_df)
                print(f"✓ 已加载原始值: {orig_csv}")
            except Exception as e:
                print(f"✗ 加载原始值失败: {orig_csv}, 错误: {e}")
                
                # 尝试加载SHP文件
                orig_shp_parts = glob.glob(os.path.join(year_dir, f"shap_original_{year}_*.shp"))
                if orig_shp_parts:
                    try:
                        # 加载每个部分并合并
                        orig_parts = [gpd.read_file(shp) for shp in orig_shp_parts]
                        merged_orig = pd.concat(orig_parts, axis=1)
                        # 删除重复的基础列
                        for col in merged_orig.columns:
                            if merged_orig.columns.to_list().count(col) > 1:
                                merged_orig = merged_orig.loc[:,~merged_orig.columns.duplicated()]
                        all_years_original.append(merged_orig)
                        print(f"✓ 已加载原始值SHP: {', '.join(orig_shp_parts)}")
                    except Exception as e2:
                        print(f"✗ 加载原始值SHP失败, 错误: {e2}")
        
        # 加载统计值CSV
        stats_csv = os.path.join(year_dir, f"shap_stats_{year}.csv")
        if os.path.exists(stats_csv):
            try:
                stats_df = pd.read_csv(stats_csv)
                # 将geometry列转换为WKT字符串表示
                if 'geometry' in stats_df.columns:
                    stats_df['geometry'] = stats_df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
                    stats_df = gpd.GeoDataFrame(stats_df, geometry='geometry')
                all_years_stats.append(stats_df)
                print(f"✓ 已加载统计值: {stats_csv}")
            except Exception as e:
                print(f"✗ 加载统计值失败: {stats_csv}, 错误: {e}")
                
                # 尝试加载SHP文件
                stats_shp_parts = glob.glob(os.path.join(year_dir, f"shap_stats_{year}_*.shp"))
                if stats_shp_parts:
                    try:
                        # 加载每个部分并合并
                        stats_parts = [gpd.read_file(shp) for shp in stats_shp_parts]
                        merged_stats = pd.concat(stats_parts, axis=1)
                        # 删除重复的基础列
                        for col in merged_stats.columns:
                            if merged_stats.columns.to_list().count(col) > 1:
                                merged_stats = merged_stats.loc[:,~merged_stats.columns.duplicated()]
                        all_years_stats.append(merged_stats)
                        print(f"✓ 已加载统计值SHP: {', '.join(stats_shp_parts)}")
                    except Exception as e2:
                        print(f"✗ 加载统计值SHP失败, 错误: {e2}")
    
    if not all_years_original or not all_years_stats:
        print("⚠️ 未能成功加载任何年度SHAP数据")
        return
    
    # 3. 合并所有年份数据
    all_orig_data = pd.concat(all_years_original, ignore_index=True)
    all_stats_data = pd.concat(all_years_stats, ignore_index=True)
    
    # 4. 按ID分组聚合
    unique_ids = all_orig_data['ID'].unique()
    
    # 准备方向和强度聚合结果
    direction_rows = []  # 方向聚合 (ma值)
    strength_rows = []   # 强度聚合 (ms值)
    
    for unique_id in tqdm(unique_ids, desc="聚合位置SHAP值"):
        # 获取该ID的所有年份数据
        id_orig = all_orig_data[all_orig_data['ID'] == unique_id]
        id_stats = all_stats_data[all_stats_data['ID'] == unique_id]
        
        if len(id_orig) == 0 or len(id_stats) == 0:
            continue
        
        # 获取代表性几何形状
        id_geometry = id_orig.iloc[0]['geometry']
        
        # === 方向聚合 ===
        dir_row = {'ID': unique_id, 'geometry': id_geometry}
        
        # 添加非对子变量原始值的平均
        valid_non_pairs = get_valid_features(NON_PAIR_FEATURES, id_orig.columns)
        for col in valid_non_pairs:
            dir_row[col] = id_orig[col].mean()
        
        # 添加对子ma值的平均
        ma_cols = [col for col in id_stats.columns if col.endswith('ma') and col != 'totalma']
        for col in ma_cols:
            dir_row[col] = id_stats[col].mean()
        
        # 添加totalma
        if 'totalma' in id_stats.columns:
            dir_row['totalma'] = id_stats['totalma'].mean()
        
        direction_rows.append(dir_row)
        
        # === 强度聚合 ===
        str_row = {'ID': unique_id, 'geometry': id_geometry}
        
        # 添加非对子变量绝对值的平均
        # 添加非对子变量绝对值的平均
        valid_non_pairs = get_valid_features(NON_PAIR_FEATURES, id_orig.columns)
        for col in valid_non_pairs:
            str_row[col] = id_orig[col].abs().mean()
        
        # 添加对子ms值的平均
        ms_cols = [col for col in id_stats.columns if col.endswith('ms')]
        for col in ms_cols:
            str_row[col] = id_stats[col].mean()
        
        # 添加total
        if 'total' in id_orig.columns:
            str_row['total'] = id_orig['total'].mean()
        
        strength_rows.append(str_row)
    
    # 5. 创建GeoDataFrame
    direction_gdf = gpd.GeoDataFrame(direction_rows, geometry='geometry')
    strength_gdf = gpd.GeoDataFrame(strength_rows, geometry='geometry')
    
    # 6. 保存结果CSV
    dir_csv_path = os.path.join(agg_dir, "shap_direction_aggregated.csv")
    str_csv_path = os.path.join(agg_dir, "shap_strength_aggregated.csv")
    
    direction_gdf.to_csv(dir_csv_path, index=False)
    strength_gdf.to_csv(str_csv_path, index=False)
    print(f"✓ 方向聚合保存为CSV: {dir_csv_path}")
    print(f"✓ 强度聚合保存为CSV: {str_csv_path}")
    
    # 7. 保存结果SHP
    dir_shp_path = os.path.join(agg_dir, "shap_direction_aggregated.shp")
    str_shp_path = os.path.join(agg_dir, "shap_strength_aggregated.shp")
    
    try:
        direction_gdf.to_file(dir_shp_path, driver="ESRI Shapefile", encoding='UTF-8')
        print(f"✓ 方向聚合SHAP保存为: {dir_shp_path}")
    except Exception as e:
        print(f"✗ 方向聚合SHAP保存失败: {e}")
        if len(direction_gdf.columns) > 255:
            split_and_save_shapefile(direction_gdf, dir_shp_path.replace('.shp', ''))
    
    try:
        strength_gdf.to_file(str_shp_path, driver="ESRI Shapefile", encoding='UTF-8')
        print(f"✓ 强度聚合SHAP保存为: {str_shp_path}")
    except Exception as e:
        print(f"✗ 强度聚合SHAP保存失败: {e}")
        if len(strength_gdf.columns) > 255:
            split_and_save_shapefile(strength_gdf, str_shp_path.replace('.shp', ''))

def calculate_climate_stats(shap_values, id_year_mapping, feature_columns, gdf_full, output_dir):
    """专门针对气候变量对计算统计信息并保存到子文件夹"""
    print("计算气候变量统计信息...")
    
    # 创建气候统计专用目录
    climate_dir = os.path.join(output_dir, "climate_stats")
    os.makedirs(climate_dir, exist_ok=True)
    
    # 将3D SHAP值转换为2D
    if len(shap_values.shape) == 3:
        shap_combined = np.sum(shap_values, axis=1)
    else:
        # 修复：直接使用 shap_values 的形状
        if len(shap_values.shape) > 2:
            # 假设 shap_values 形状为 [samples, seq_len, features]
            seq_len = shap_values.shape[1]
            feature_dim = shap_values.shape[2]
            shap_combined = np.sum(shap_values.reshape(-1, seq_len, feature_dim), axis=1)
        else:
            # 已经是2D形状
            shap_combined = shap_values
    
    # 创建基础DataFrame
    climate_data = {
        'ID': id_year_mapping['ID'].values,
        'year': id_year_mapping['year'].values,
        'geometry': gdf_full.loc[id_year_mapping.index, 'geometry'].values
    }
    
    # 将所有气候变量添加到DataFrame
    feature_indices = {name: idx for idx, name in enumerate(feature_columns)}
    climate_variables = set()
    
    for base, derived in CLIMATE_PAIRS:
        if base in feature_indices and derived in feature_indices:
            base_idx = feature_indices[base]
            derived_idx = feature_indices[derived]
            
            # 添加原始SHAP值
            climate_data[base] = shap_combined[:, base_idx]
            climate_data[derived] = shap_combined[:, derived_idx]
            
            # 记录气候变量名称
            climate_variables.add(base)
            climate_variables.add(derived)
    
    # 创建GeoDataFrame
    climate_gdf = gpd.GeoDataFrame(climate_data, geometry='geometry')
    # 检查是否找到任何气候变量
    if not climate_variables:
        print("警告: 未找到任何气候变量，生成空的气候统计")
        # 创建最小结果并返回
        min_result = gpd.GeoDataFrame({
            'ID': id_year_mapping['ID'].values,
            'year': id_year_mapping['year'].values,
            'geometry': gdf_full.loc[id_year_mapping.index, 'geometry'].values,
            'empty': np.ones(len(id_year_mapping))  # 标记为空
        }, geometry='geometry')
        
        # 保存最小报告
        min_csv_path = os.path.join(climate_dir, "climate_variables_empty.csv")
        min_result.to_csv(min_csv_path, index=False)
        print(f"✓ 生成空气候统计报告: {min_csv_path}")
        
        return min_result
    # 计算气候变量对的ma和ms值
    ma_values = {}
    ms_values = {}
    
    for base, derived in CLIMATE_PAIRS:
        if base in climate_gdf.columns and derived in climate_gdf.columns:
            # 方向值 (ma)
            ma_values[f"{base}ma"] = (climate_gdf[base] + climate_gdf[derived]) / 2
            
            # 强度值 (ms)
            ms_values[f"{base}ms"] = (np.abs(climate_gdf[base]) + np.abs(climate_gdf[derived])) / 2
    
    # 添加统计列
    climate_gdf = climate_gdf.assign(**ma_values)
    climate_gdf = climate_gdf.assign(**ms_values)
    
    # 计算气候总和列
    climate_gdf['total'] = climate_gdf[[col for col in climate_variables if col in climate_gdf.columns]].sum(axis=1)
    climate_gdf['matotal'] = climate_gdf[[col for col in ma_values.keys()]].sum(axis=1)
    climate_gdf['mstotal'] = climate_gdf[[col for col in ms_values.keys()]].sum(axis=1)
    
    # 保存CSV文件
    csv_path = os.path.join(climate_dir, "climate_variables_stats.csv")
    climate_gdf.to_csv(csv_path, index=False)
    print(f"✓ 气候变量统计CSV已保存: {csv_path}")
    
    # 保存SHP文件
    shp_base = os.path.join(climate_dir, "climate_variables_stats")
    try:
        climate_gdf.to_file(f"{shp_base}.shp", driver="ESRI Shapefile", encoding='UTF-8')
        print(f"✓ 气候变量统计SHP已保存: {shp_base}.shp")
    except Exception as e:
        print(f"✗ 气候变量统计SHP保存失败: {e}")
        # 如果字段过多，分割保存
        split_and_save_shapefile(climate_gdf, shp_base, max_fields=250)
    
    # 为各个年份单独保存气候统计
    unique_years = climate_gdf['year'].unique()
    for year in unique_years:
        year_climate = climate_gdf[climate_gdf['year'] == year].copy()
        
        # 保存CSV
        year_csv_path = os.path.join(climate_dir, f"climate_stats_{year}.csv")
        year_climate.to_csv(year_csv_path, index=False)
        print(f"✓ {year}年气候统计CSV已保存: {year_csv_path}")
        
        # 保存SHP
        year_shp_base = os.path.join(climate_dir, f"climate_stats_{year}")
        try:
            year_climate.to_file(f"{year_shp_base}.shp", driver="ESRI Shapefile", encoding='UTF-8')
            print(f"✓ {year}年气候统计SHP已保存: {year_shp_base}.shp")
        except Exception as e:
            print(f"✗ {year}年气候统计SHP保存失败: {e}")
            split_and_save_shapefile(year_climate, year_shp_base, max_fields=250)
    
    # 创建气候变量重要性可视化 - 修复了append问题
    climate_importance_data = []  # 使用列表收集数据
    for col in climate_variables:
        if col in climate_gdf.columns:
            importance = np.mean(np.abs(climate_gdf[col]))
            climate_importance_data.append({
                'variable': col,
                'importance': importance
            })
    
    # 从列表创建DataFrame
    climate_importance = pd.DataFrame(climate_importance_data)
    
    # 排序并可视化
    if not climate_importance.empty:
        climate_importance = climate_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        plt.barh(climate_importance['variable'], climate_importance['importance'])
        plt.xlabel('Mean |SHAP| Value')
        plt.ylabel('Climate Variable')
        plt.title('Climate Variables Importance')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(climate_dir, "climate_importance.png"), dpi=1200)
        plt.close()
        
        # 保存CSV
        climate_importance.to_csv(os.path.join(climate_dir, "climate_importance.csv"), index=False)
    
    print(f"✓ 气候变量分析完成，结果保存在: {climate_dir}")
    
    return climate_gdf
def generate_global_shap_file(shap_values, X_all, id_year_mapping, feature_columns_adjusted, gdf_full, output_dir):
    """生成基于全部样本的全局SHAP文件 - 修改版"""
    print("生成全局SHAP文件 (基于所有样本)...")
    
    # 创建全局SHAP专用目录
    global_dir = os.path.join(output_dir, "global_shap")
    os.makedirs(global_dir, exist_ok=True)
    
    # 处理SHAP值 (将3D转为2D)
    if len(shap_values.shape) == 3:
        shap_combined = np.sum(shap_values, axis=1)
    else:
        seq_len = X_all.shape[1]
        feature_dim = X_all.shape[2]
        shap_combined = shap_values.reshape(-1, seq_len, feature_dim)
        shap_combined = np.sum(shap_combined, axis=1)
    
    # ========== 创建完整的全局SHAP DataFrame ==========
    # 基础列
    global_data = {
        'ID': id_year_mapping['ID'].values,
        'year': id_year_mapping['year'].values,
        'ID_year': id_year_mapping['ID_year'].values,
        'geometry': gdf_full.loc[id_year_mapping.index, 'geometry'].values
    }
    
    # 添加所有SHAP原始值
    for i, col_name in enumerate(feature_columns_adjusted):
        if i < shap_combined.shape[1]:
            global_data[col_name] = shap_combined[:, i]
    
    # 创建全局GeoDataFrame
    global_gdf = gpd.GeoDataFrame(global_data, geometry='geometry')
    
    # ========== 计算与添加统计列 ==========
    # 计算对子变量的ma和ms值
    ma_values = {}
    ms_values = {}
    
    for base, derived in ALL_PAIRS:
        if base in global_gdf.columns and derived in global_gdf.columns:
            # 方向值 (ma)
            ma_values[f"{base}ma"] = (global_gdf[base] + global_gdf[derived]) / 2
            
            # 强度值 (ms)
            ms_values[f"{base}ms"] = (np.abs(global_gdf[base]) + np.abs(global_gdf[derived])) / 2
    
    # 添加所有ma和ms列
    global_gdf = global_gdf.assign(**ma_values)
    global_gdf = global_gdf.assign(**ms_values)
    
    # 计算total列 (所有特征的综合影响)
    # 对非对子变量计算绝对值，仅使用存在的非对子变量
    non_pair_abs_sum = np.zeros(len(global_gdf))
    valid_non_pairs = get_valid_features(NON_PAIR_FEATURES, global_gdf.columns)
    if valid_non_pairs:
        for col in valid_non_pairs:
            non_pair_abs_sum += np.abs(global_gdf[col].values)
    else:
        print("警告: 计算total时未找到任何非对子变量")
        
    # 对所有对子变量计算绝对值均值
    pair_abs_mean_sum = np.zeros(len(global_gdf))
    for base, derived in ALL_PAIRS:
        if base in global_gdf.columns and derived in global_gdf.columns:
            pair_abs = (np.abs(global_gdf[base].values) + np.abs(global_gdf[derived].values)) / 2
            pair_abs_mean_sum += pair_abs
    
    # 添加total列
    global_gdf['total'] = non_pair_abs_sum + pair_abs_mean_sum
    
    # 计算totalma列，仅使用存在的非对子变量
    non_pair_sum = np.zeros(len(global_gdf))
    if valid_non_pairs:  # 重用前面获取的列表
        for col in valid_non_pairs:
            non_pair_sum += global_gdf[col].values
    else:
        print("警告: 计算totalma时未找到任何非对子变量")
    
    ma_sum = np.zeros(len(global_gdf))
    for key in ma_values.keys():
        ma_sum += global_gdf[key].values
    
    # 添加totalma列
    global_gdf['totalma'] = non_pair_sum + ma_sum
    
    # ========== 保存结果 ==========
    # 1. 保存CSV文件
    csv_path = os.path.join(global_dir, "global_all_samples_shap.csv")
    global_gdf.to_csv(csv_path, index=False)
    print(f"✓ 全局SHAP CSV已保存: {csv_path}")
    
    # 2. 分割保存SHP文件
    shp_base = os.path.join(global_dir, "global_all_samples_shap")
    split_and_save_shapefile(global_gdf, shp_base, max_fields=250)

def split_and_save_shapefile(gdf, base_path, max_fields=250):
    """将大型GeoDataFrame分割成多个Shapefile，增加序号后缀"""
    # 基本字段
    base_cols = ['ID', 'geometry']
    if 'year' in gdf.columns:
        base_cols.append('year')
    
    # 其他字段
    other_cols = [col for col in gdf.columns if col not in base_cols]
    
    # 计算需要分几个部分
    num_parts = (len(other_cols) + max_fields - 1) // max_fields
    
    # 分批处理字段
    for i in range(num_parts):
        start_idx = i * max_fields
        end_idx = min((i+1) * max_fields, len(other_cols))
        part_cols = base_cols + other_cols[start_idx:end_idx]
        part_df = gdf[part_cols].copy()
        part_path = f"{base_path}_{i+1}.shp"
        
        try:
            part_df.to_file(part_path, driver="ESRI Shapefile", encoding='UTF-8')
            print(f"✓ 部分{i+1}保存为: {part_path}")
        except Exception as e:
            print(f"✗ 部分{i+1}保存失败: {part_path}, 错误: {e}")
# 可视化SHAP重要性 - 优化图像尺寸以适应所有特征
def analyze_shap_importance(shap_values_test, seq_len, feature_dim_per_year, X_cols):
    """分析全局特征重要性 - 计算非对子变量和对子变量的重要性 - 修改版"""
    # 处理SHAP值维度不变...
    if len(shap_values_test.shape) == 3:  
        combined_shap = np.sum(shap_values_test, axis=1)
    else:
        try:
            if shap_values_test.shape[1] == feature_dim_per_year:
                combined_shap = shap_values_test
            else:
                reshaped = shap_values_test.reshape(-1, seq_len, feature_dim_per_year)
                combined_shap = np.sum(reshaped, axis=1)
        except Exception as e:
            print(f"SHAP形状处理错误: {e}")
            print(f"SHAP值形状: {shap_values_test.shape}")
            combined_shap = shap_values_test
    
    # 计算平均绝对SHAP值
    shap_abs_mean = np.mean(np.abs(combined_shap), axis=0)
    
    # 对原始特征计算平均SHAP值
    shap_mean = np.mean(combined_shap, axis=0)
    
    # 创建原始特征的平均绝对SHAP值数据框
    df_original = pd.DataFrame({
        'feature_index': np.arange(feature_dim_per_year),
        'feature_name': X_cols[:feature_dim_per_year],
        'mean_abs_shap': shap_abs_mean
    })
    
    # ========== 计算对子变量的重要性 ==========
    # 存储对子变量的重要性数据
    pair_importance = []
    
    # 索引映射，将特征名映射到索引
    feature_to_idx = {name: idx for idx, name in enumerate(X_cols) if idx < len(shap_abs_mean)}
    
    # 计算每对对子变量的重要性
    for base, derived in ALL_PAIRS:
        # 检查两个特征是否在索引映射中
        if base in feature_to_idx and derived in feature_to_idx:
            base_idx = feature_to_idx[base]
            derived_idx = feature_to_idx[derived]
            
            # 计算方向平均值 (MA)
            ma_value = (shap_mean[base_idx] + shap_mean[derived_idx]) / 2
            
            # 计算强度平均值 (MS)
            ms_value = (shap_abs_mean[base_idx] + shap_abs_mean[derived_idx]) / 2
            
            # 添加到重要性列表
            pair_importance.append({
                'pair': f"{base}-{derived}",
                'base': base,
                'derived': derived,
                'ma_value': ma_value,
                'ms_value': ms_value,
                'mean_abs_shap': ms_value  # 用于排序
            })
    
    # 创建变量对重要性DataFrame
    df_pairs = pd.DataFrame(pair_importance)
    if not df_pairs.empty:
        df_pairs['feature_name'] = df_pairs['base'] + 'ms'  # 使用base+'ms'作为特征名
    
    # ========== 创建用于可视化的DataFrame ==========
    # 筛选非对子变量
    # 筛选非对子变量，只使用存在的非对子变量
    valid_non_pairs = [col for col in NON_PAIR_FEATURES if col in X_cols[:feature_dim_per_year]]
    valid_non_pairs = get_valid_features(NON_PAIR_FEATURES, X_cols[:feature_dim_per_year])
    df_non_pairs = df_original[df_original['feature_name'].isin(valid_non_pairs)].copy()

    # 合并非对子变量和对子变量强度值
    if not df_pairs.empty:
        if len(df_non_pairs) > 0:  # 有非对子变量时
            df_for_viz = pd.concat([
                df_non_pairs,
                df_pairs[['feature_name', 'mean_abs_shap']].copy()
            ], ignore_index=True)
        else:  # 没有非对子变量时只使用对子变量
            df_for_viz = df_pairs[['feature_name', 'mean_abs_shap']].copy()
    else:
        df_for_viz = df_non_pairs.copy()
    
    # 按重要性排序
    df_for_viz = df_for_viz.sort_values('mean_abs_shap', ascending=False)
    
    # 创建完整特征重要性DataFrame (包含所有原始特征)
    df_all = df_original.copy()
    df_all = df_all.sort_values('mean_abs_shap', ascending=False)
    
    # 准备用于可视化的shap_abs_mean数组
    shap_abs_mean_viz = np.array(df_for_viz['mean_abs_shap'])
    
    # 调试信息
    print(f"非对子变量数量: {len(df_non_pairs)}")
    if not df_pairs.empty:
        print(f"对子变量类数量: {len(df_pairs)}")
    print(f"可视化变量总数: {len(df_for_viz)}")
    
    return df_for_viz, df_all, shap_abs_mean_viz


def generate_yearly_shap_importance(shap_values, X_all, id_year_mapping, feature_columns_adjusted, output_dir):
    """为每年生成特征重要性图 - 修改版"""
    print("生成年度特征重要性图...")
    
    # 创建年度重要性专用目录
    yearly_imp_dir = os.path.join(output_dir, "yearly_importance")
    os.makedirs(yearly_imp_dir, exist_ok=True)
    
    # 获取唯一年份并排序
    unique_years = sorted(id_year_mapping['year'].unique())
    
    # 所有年份的TOP特征重要性
    yearly_top_features = {}
    
    for year in tqdm(unique_years, desc="生成年度特征重要性"):
        # 获取该年份的样本索引
        year_mask = id_year_mapping['year'] == year
        year_indices = np.where(year_mask)[0]
        
        if len(year_indices) == 0:
            print(f"警告: {year}年没有样本")
            continue
        
        # 创建年份目录
        year_dir = os.path.join(yearly_imp_dir, f"year_{year}")
        os.makedirs(year_dir, exist_ok=True)
        
        # 获取该年份的SHAP值
        year_shap = shap_values[year_indices]
        
        try:
            # 分析该年的特征重要性
            seq_len = X_all.shape[1]
            feature_dim = X_all.shape[2]
            df_for_viz, df_all, shap_abs_mean_viz = analyze_shap_importance(
                year_shap, seq_len, feature_dim, feature_columns_adjusted
            )
            
            # 保存TOP30特征用于后续趋势分析
            yearly_top_features[year] = df_for_viz.head(30).copy()
            
            # 可视化该年的特征重要性
            visualize_shap_importance(
                df_for_viz, df_all, shap_abs_mean_viz,
                year_dir, prefix=f"year_{year}"
            )
            
            print(f"✓ {year}年特征重要性分析完成")
        except Exception as e:
            print(f"✗ {year}年特征重要性分析失败: {e}")
    
    # 创建跨年份的特征重要性趋势图
    if yearly_top_features:
        try:
            generate_importance_trends(yearly_top_features, yearly_imp_dir)
        except Exception as e:
            print(f"生成特征重要性趋势图失败: {e}")

def generate_importance_trends(yearly_top_features, output_dir):
    """生成跨年份的特征重要性趋势图"""
    print("生成特征重要性趋势图...")
    
    # 提取所有年份中出现频率最高的特征
    all_features = {}
    for year, df in yearly_top_features.items():
        for _, row in df.iterrows():
            feature = row['feature_name']
            if feature not in all_features:
                all_features[feature] = 0
            all_features[feature] += 1
    
    # 获取出现频率最高的前15个特征
    top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:15]
    top_feature_names = [f[0] for f in top_features]
    
    # 收集这些特征在每年的重要性值
    trends_data = {}
    years = sorted(yearly_top_features.keys())
    
    for feature in top_feature_names:
        trends_data[feature] = []
        for year in years:
            df = yearly_top_features[year]
            feature_row = df[df['feature_name'] == feature]
            if not feature_row.empty:
                trends_data[feature].append(feature_row['mean_abs_shap'].values[0])
            else:
                # 如果该年没有这个特征，用0填充
                trends_data[feature].append(0)
    
    # 绘制趋势图
    plt.figure(figsize=(15, 10))
    
    for feature in top_feature_names:
        plt.plot(years, trends_data[feature], marker='o', label=feature)
    
    plt.xlabel('Year')
    plt.ylabel('Mean |SHAP| Value')
    plt.title('Top Features Importance Trends Across Years')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图表
    trends_path = os.path.join(output_dir, "feature_importance_trends.png")
    plt.savefig(trends_path, dpi=1200)
    plt.close()
    
    print(f"✓ 特征重要性趋势图已保存: {trends_path}")
    
    # 保存趋势数据为CSV
    trends_df = pd.DataFrame(trends_data, index=years)
    trends_csv = os.path.join(output_dir, "feature_importance_trends.csv")
    trends_df.to_csv(trends_csv)
    print(f"✓ 特征重要性趋势数据已保存: {trends_csv}")
def visualize_shap_importance(df_shap_importance, df_all_importance, shap_abs_mean_viz, 
                             output_dir, prefix="global"):
    """可视化全局特征重要性 - 仅展示非气候对变量和气候差值变量"""
    # 保存完整CSV
    df_all_importance.to_csv(os.path.join(output_dir, f"{prefix}_shap_global_importance_all.csv"), index=False)
    
    # 保存用于可视化的CSV
    df_shap_importance.to_csv(os.path.join(output_dir, f"{prefix}_shap_global_importance_viz.csv"), index=False)
    
    # 全局特征重要性条形图
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(shap_abs_mean_viz)), shap_abs_mean_viz)
    plt.xlabel("Feature Index")
    plt.ylabel("Mean |SHAP|")
    plt.title(f"{prefix.capitalize()} Feature Importance")
    plt.savefig(os.path.join(output_dir, f"{prefix}_shap_global_importance.png"), dpi=1200)
    plt.close()
    
    # 展示所有特征（非气候对变量和气候差值变量）- 动态调整图像大小
    n_features = len(df_shap_importance)
    
    # 动态计算图像高度 - 根据特征数量调整
    fig_height = max(10, n_features * 0.25)
    plt.figure(figsize=(14, fig_height))
    
    y_pos = np.arange(n_features)
    plt.barh(y_pos, df_shap_importance['mean_abs_shap'], align='center')
    plt.yticks(y_pos, df_shap_importance['feature_name'], fontsize=8)
    plt.xlabel('Mean |SHAP| Value')
    plt.title('Feature Importance Ranking (Non-Climate Pairs + Climate Strength)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_features_importance.png"), dpi=1200)
    plt.close()
    
    # 单独展示气候强度变量(ms)
    ms_features = df_shap_importance[df_shap_importance['feature_name'].str.endswith('ms')]
    if not ms_features.empty:
        fig_height_ms = max(8, len(ms_features) * 0.3)
        plt.figure(figsize=(14, fig_height_ms))
        
        y_pos_ms = np.arange(len(ms_features))
        plt.barh(y_pos_ms, ms_features['mean_abs_shap'], align='center')
        plt.yticks(y_pos_ms, ms_features['feature_name'], fontsize=9)
        plt.xlabel('Mean |SHAP| Value')
        plt.title('Climate Strength (ms) Importance Ranking')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "climate_strength_importance.png"), dpi=1200)
        plt.close()
    
    # 如果特征数量过多，可以分成多个图展示
    if n_features > 100:
        # 每个图显示50个特征
        chunk_size = 40
        for i in range(0, n_features, chunk_size):
            end_idx = min(i + chunk_size, n_features)
            subset = df_shap_importance.iloc[i:end_idx]
            
            plt.figure(figsize=(14, 12))
            y_pos = np.arange(len(subset))
            plt.barh(y_pos, subset['mean_abs_shap'], align='center')
            plt.yticks(y_pos, subset['feature_name'], fontsize=9)
            plt.xlabel('Mean |SHAP| Value')
            plt.title(f'Feature Importance Ranking (Features {i+1}-{end_idx})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"features_importance_{i+1}_to_{end_idx}.png"), dpi=1200)
            plt.close()

# SHAP分析主函数
def comprehensive_shap_analysis(model, X_all, id_year_mapping, feature_columns, gdf_full, sess, output_dir):
    """基于全部样本的完整SHAP分析函数 - 修改版"""
    # 声明全局变量 - 添加到函数开始
    global NON_PAIR_FEATURES
    try:
        print("开始基于全部样本的SHAP分析...")
        print(f"模型输入形状: {model.input_shape}")
        print(f"X_all形状: {X_all.shape}")
        print(f"特征列数量: {len(feature_columns)}")
        
        # 检查调整特征维度的代码不变...
        expected_features = model.input_shape[-1]
        actual_features = X_all.shape[-1]
        
        # 特征比较不变...
        print("\n特征比较:")
        print("模型训练使用的特征:")
        print("\n".join(f"{i}: {col}" for i, col in enumerate(feature_columns)))
        
        if actual_features != expected_features:
            # 假设额外特征没有名称，我们用占位符表示
            all_features = feature_columns + [f"unknown_{i}" for i in range(len(feature_columns), actual_features)]
            print("\nSHAP分析使用的特征 (包含额外特征):")
            print("\n".join(f"{i}: {col}" for i, col in enumerate(all_features)))
            
            print(f"\n⚠️ 特征维度不匹配: 模型期望{expected_features}个特征，但数据有{actual_features}个特征")
            print(f"自动调整为使用前{expected_features}个特征...")
            
            # 只使用前expected_features个特征
            X_all_adjusted = X_all[:, :, :expected_features]
            
            # 同时调整特征列列表
            if len(feature_columns) > expected_features:
                feature_columns_adjusted = feature_columns[:expected_features]
                print(f"特征列表已从{len(feature_columns)}个调整为{len(feature_columns_adjusted)}个")
                # 打印被排除的特征
                excluded_features = feature_columns[expected_features:]
                print(f"被排除的特征: {excluded_features}")
            else:
                feature_columns_adjusted = feature_columns
                print("警告: 特征列表比模型期望的特征数量短，这可能导致特征名映射错误")
        else:
            X_all_adjusted = X_all
            feature_columns_adjusted = feature_columns
            print("✓ 特征维度匹配正确")
        
        # 验证变量对
        print("\n验证变量对定义...")
        valid_pairs = [(base, derived) for base, derived in ALL_PAIRS 
               if base in feature_columns_adjusted and derived in feature_columns_adjusted]
        print(f"在特征列中找到 {len(valid_pairs)}/{len(ALL_PAIRS)} 个有效变量对")
        # 验证非对子变量
        valid_non_pairs = get_valid_features(NON_PAIR_FEATURES, feature_columns_adjusted)
        print(f"在特征列中找到 {len(valid_non_pairs)}/{len(NON_PAIR_FEATURES)} 个有效非对子变量")
        if valid_non_pairs:
            print(f"非对子变量: {', '.join(valid_non_pairs)}")
        else:
            print("警告: 未找到任何非对子变量，将使用空列表")
            
            NON_PAIR_FEATURES = []
        # 显示前几个对子
        if valid_pairs:
            print("示例对子:")
            for i, (base, derived) in enumerate(valid_pairs[:5]):
                print(f"  {i+1}. {base} - {derived}")
        
        
        
        # 创建SHAP解释器不变...
        explainer, input_ph = create_shap_explainer(
            model, 
            X_all_adjusted, 
            sess, 
            use_kmeans=True, 
            n_clusters=N_CLUSTERS
        )
        
        # 计算所有样本的SHAP值不变...
        shap_values_all = calculate_all_shap_values(explainer, X_all_adjusted, input_ph)
        
        # 维度信息不变...
        seq_len = X_all_adjusted.shape[1]
        feature_dim_per_year = X_all_adjusted.shape[2]
        
        # 全局特征重要性分析
        print("分析全局特征重要性...")
        global_imp_dir = os.path.join(output_dir, "global_importance")
        os.makedirs(global_imp_dir, exist_ok=True)
        
        df_for_viz, df_all, shap_abs_mean_viz = analyze_shap_importance(
            shap_values_all, seq_len, feature_dim_per_year, feature_columns_adjusted
        )
        
        # 可视化全局SHAP重要性
        visualize_shap_importance(
            df_for_viz, df_all, shap_abs_mean_viz, 
            global_imp_dir, prefix="global"
        )
        
        # 生成年度SHAP输出
        print("生成按年份分组的SHAP输出...")
        generate_yearly_shap_outputs(
            shap_values_all, X_all_adjusted, id_year_mapping, 
            feature_columns_adjusted, gdf_full, 
            output_dir
        )
        
        # 生成年度特征重要性图表
        print("生成年度特征重要性图表...")
        generate_yearly_shap_importance(
            shap_values_all, X_all_adjusted, id_year_mapping,
            feature_columns_adjusted, output_dir
        )
        
        # 生成聚合SHAP输出
        print("生成聚合SHAP输出...")
        generate_aggregated_shap_outputs(
            os.path.join(output_dir, "yearly_shap"),
            os.path.join(output_dir, "aggregated"),
            feature_columns_adjusted
        )
        
        # 生成全局SHAP文件
        generate_global_shap_file(
            shap_values_all, X_all_adjusted, id_year_mapping, feature_columns_adjusted, 
            gdf_full, output_dir
        )
        # 添加对气候变量的专门分析
        print("分析气候变量...")
        calculate_climate_stats(
            shap_values_all, id_year_mapping, feature_columns_adjusted, 
            gdf_full, output_dir
        )
        print("SHAP分析完成!")
        
    except Exception as e:
        print(f"SHAP分析出错: {e}")
        traceback.print_exc()

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

def main():
    print("\n===== Starting TimesNet Analysis =====")
    save_current_script(OUTPUT_DIR)
    print(f"✅ 当前脚本已保存为: {OUTPUT_DIR}/backup_timesnet_script.py")
    
    # 定义关键文件路径
    model_path = os.path.join(OUTPUT_DIR, "models", "timesnet_final_model.h5")
    scaler_X_path = os.path.join(OUTPUT_DIR, "models", "scaler_X.pkl")
    scaler_y_path = os.path.join(OUTPUT_DIR, "models", "scaler_y.pkl")
    train_memmap_path = os.path.join(OUTPUT_DIR, "models", "X_train_memmap.dat")
    test_memmap_path = os.path.join(OUTPUT_DIR, "models", "X_test_memmap.dat")
    shape_info_path = os.path.join(OUTPUT_DIR, "models", "shape_info.json")
    id_year_mapping_path = os.path.join(OUTPUT_DIR, "models", "id_year_mapping.pkl")
    x_scaled_path = os.path.join(OUTPUT_DIR, "models", "X_scaled.pkl")  # 新增: X_scaled保存路径
    
    # 检查模型和标准化器是否已存在
    model_exists = os.path.exists(model_path)
    scalers_exist = os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)
    
    # 加载数据
    all_files = load_data_files()
    print(f"Loaded {len(all_files)} data files")
    
    try:
        if not (model_exists and scalers_exist):
            # 如果模型或标准化器不存在，执行完整训练流程
            print("模型或标准化器不存在，执行完整训练流程...")
            
            # 预处理数据 - 注意: 接收id_year_mapping作为返回值
            X_scaled, y_scaled, pixel_ids, geo_info, feature_columns, scaler_X, scaler_y, gdf_full, id_year_mapping = preprocess_data(all_files)
            
            # 保存标准化器和ID-年份映射
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)
            joblib.dump(id_year_mapping, id_year_mapping_path)
            
            # 保存X_scaled用于后续SHAP分析 - 新增
            print("保存X_scaled数据用于后续分析...")
            if X_scaled.nbytes > 1e9:  # 如果大于1GB
                print("X_scaled数据较大，使用内存映射保存...")
                X_scaled_memmap = np.memmap(x_scaled_path, dtype='float32', mode='w+', 
                                          shape=X_scaled.shape)
                X_scaled_memmap[:] = X_scaled[:]
                X_scaled_memmap.flush()
                print(f"✅ X_scaled已保存为内存映射: {x_scaled_path}")
            else:
                joblib.dump(X_scaled, x_scaled_path)
                print(f"✅ X_scaled已保存: {x_scaled_path}")
            
            print(f"✅ 标准化器和ID-年份映射已保存")
            
            # 训练模型
            print("\n===== Training TimesNet Model =====")
            model, X_test_windows, y_test_targets, history = train_and_evaluate_model(X_scaled, y_scaled, feature_columns)
            plot_training_history(history, OUTPUT_DIR)
            
            # 评估模型
            print("\n===== Evaluating Model Performance =====")
            y_pred, evaluation_metrics = evaluate_model_performance(model, X_test_windows, y_test_targets, scaler_y, OUTPUT_DIR)
            
            # 拆分训练和测试数据，准备SHAP分析
            num_samples = X_scaled.shape[0]
            train_size = int(0.8 * num_samples)
            X_train_s = X_scaled[:train_size]
            X_test_s = X_scaled[train_size:]
            idx_test = np.arange(train_size, num_samples)
            
            # 创建内存映射文件
            print("创建内存映射文件...")
            X_train_memmap = create_memmap_features(X_train_s, train_memmap_path)
            X_test_memmap = create_memmap_features(X_test_s, test_memmap_path)
            
            # 保存形状和索引信息
            seq_len = SEQ_LEN
            feature_dim_per_year = X_train_s.shape[2]
            
            shape_info = {
                'feature_dim_per_year': int(feature_dim_per_year),
                'seq_len': int(seq_len),
                'idx_test': idx_test.tolist(),
                'X_cols': feature_columns,
                'train_shape': [int(dim) for dim in X_train_s.shape],
                'test_shape': [int(dim) for dim in X_test_s.shape],
                'x_scaled_shape': [int(dim) for dim in X_scaled.shape]  # 添加X_scaled形状信息
            }
            
            with open(shape_info_path, 'w') as f:
                json.dump(shape_info, f)
            
            print(f"✅ 形状和索引信息已保存到: {shape_info_path}")
            
        else:
            # 如果模型和标准化器已存在，加载它们
            print("加载现有模型和标准化器...")
            model = tf.keras.models.load_model(model_path, custom_objects={'huber_loss': huber_loss, 'kge': kge})
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
            
            # 加载形状和索引信息
            with open(shape_info_path, 'r') as f:
                shape_info = json.load(f)
            
            feature_columns = shape_info['X_cols']
            seq_len = shape_info['seq_len']
            feature_dim_per_year = shape_info['feature_dim_per_year']
            idx_test = np.array(shape_info['idx_test'])
            
            # 检查是否存在ID-年份映射，如果不存在则重新创建
            if os.path.exists(id_year_mapping_path):
                id_year_mapping = joblib.load(id_year_mapping_path)
                print("✅ 已加载ID-年份映射")
            else:
                print("⚠️ 未找到ID-年份映射，重新处理数据...")
                # 创建一个简单的预处理函数，只返回id_year_mapping
                _, _, _, _, _, _, _, _, id_year_mapping = preprocess_data(all_files)
                # 保存映射以备后用
                joblib.dump(id_year_mapping, id_year_mapping_path)
            
            # ===== 关键修复: 加载或重新计算X_scaled =====
            if os.path.exists(x_scaled_path):
                print("加载保存的X_scaled数据...")
                try:
                    # 尝试直接加载
                    X_scaled = joblib.load(x_scaled_path)
                    print(f"✅ 已加载X_scaled数据，形状: {X_scaled.shape}")
                except:
                    # 如果失败，尝试作为内存映射加载
                    print("尝试作为内存映射加载X_scaled...")
                    if 'x_scaled_shape' in shape_info:
                        X_scaled = np.memmap(x_scaled_path, dtype='float32', mode='r', 
                                          shape=tuple(shape_info['x_scaled_shape']))
                        print(f"✅ 已加载X_scaled内存映射，形状: {X_scaled.shape}")
                    else:
                        print("⚠️ 无法确定X_scaled形状，重新计算...")
                        X_scaled, _, _, _, _, _, _, gdf_full, _ = preprocess_data(all_files)
            else:
                print("未找到保存的X_scaled数据，重新计算...")
                X_scaled, _, _, _, _, _, _, gdf_full, _ = preprocess_data(all_files)
            
            # 只加载gdf_full
            if 'gdf_full' not in locals() or gdf_full is None:
                print("处理数据以获取地理信息...")
                _, _, _, _, _, _, _, gdf_full, _ = preprocess_data(all_files)
        
        # 执行SHAP分析
        print("\n===== 执行全面SHAP归因分析 =====")
        # 获取TF会话
        sess = tf.compat.v1.keras.backend.get_session()
        
        # 使用新的comprehensive_shap_analysis
        comprehensive_shap_analysis(
            model=model,
            X_all=X_scaled,  # 使用全部样本
            id_year_mapping=id_year_mapping,
            feature_columns=feature_columns,
            gdf_full=gdf_full,
            sess=sess,
            output_dir=os.path.join(OUTPUT_DIR, "attributions")
        )
        
        print("\n===== Analysis Complete =====")
        print(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)
        
        # Start analysis
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
