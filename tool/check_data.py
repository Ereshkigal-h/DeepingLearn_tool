import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
sns.set_theme(style="whitegrid", font='SimHei')
def load_data(path,columns=True,info=False,pop:list=None):
    """
        读取 CSV 数据，并根据参数执行数据概览和列提取操作。

        参数 (Parameters):
        -----------------
        path : str
            CSV 文件的路径。
        columns : bool, 默认 True
            是否打印数据集中每一列的去重唯一值（用于快速了解离散特征的取值分布）。
        info : bool, 默认 False
            是否调用 pd.DataFrame.info() 打印数据的基本信息（内存占用、数据类型、缺失值等）。
        pop : list, 默认 None
            需要从原始数据中单独提取出来（并从原数据中删除）的列名列表。通常用于提取标签列或 ID 列。

        返回 (Returns):
        --------------
        如果 pop 不为 None:
            return pops, data  (返回一个包含提取列的新 DataFrame，以及删除这些列后的原 DataFrame)
        如果 pop 为 None:
            return data        (直接返回原始 DataFrame)
        """
    print("load data")
    try:
        data = pd.read_csv(path)
        print("load data success")
        print(f"(行，列)：{data.shape}")
        if columns:
            data_example={col: data[col].dropna().unique() for col in data.columns}
            print("列信息")
            for col in data.columns:
                print(f"{col}:{data_example[col]}")
        if info:
            print("基本信息")
            print(data.info())
        if pop is not None:
            pops =pd.DataFrame()
            for col in pop:
                pop_temp=data.pop(col)
                pops[col]=pop_temp
            return pops,data
        return data
    except FileNotFoundError:
        print("file not found")
        return None
def plot_features_frequency(df,plot_cols:list,kde=False):
    """
        1. 绘制指定特征的频率分布直方图 (Histogram + KDE)
        （注：此函数逻辑与 plot_numerical_features 高度一致，可根据业务语义选择使用）

        参数:
        -----
        df : pandas.DataFrame
            包含数据的 DataFrame。
        plot_cols : list
            需要绘制频率分布图的列名列表。
        """
    plot_cols=[col for col in plot_cols if col in df.columns]
    print(f"正在绘画频率图,绘画的列{plot_cols}")
    if not plot_cols:
        print("无内容")
        return
    n_cols = 3
    n_rows = math.ceil(len(plot_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() if len(plot_cols) > 1 else [axes]

    for i, col in enumerate(plot_cols):
        sns.histplot(df[col].dropna(), kde=kde, ax=axes[i], color='cornflowerblue', bins=30)
        axes[i].set_title(f'[{col}] 数值分布', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('频数')
        axes[i].set_xlabel('特征取值')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
def plot_correlation_heatmap(df, num_cols):
    """
        3. 绘制数值特征的相关性热力图 (Correlation Heatmap)
        核心目的：发现特征之间是否高度冗余 (多重共线性)，以及特征与预测目标之间的相关性。
        需要先进行数值化
        参数:
        -----
        df : pandas.DataFrame
            包含数据的 DataFrame。
        num_cols : list
            需要计算相关性的数值型特征列名列表。
        """
    valid_cols = [col for col in num_cols if col in df.columns]
    if len(valid_cols) < 2:
        print("[热力图] 至少需要 2 个数值列才能计算相关性！")
        return

    print(f"正在绘制 {len(valid_cols)} 个特征的相关性热力图...")

    plt.figure(figsize=(10, 8))
    # 计算皮尔逊相关系数矩阵
    corr_matrix = df[valid_cols].corr()

    # 绘制热力图：annot=True显示具体数字，cmap设定红蓝冷暖色调
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, linewidths=.5, cbar_kws={"shrink": .8})

    plt.title('数值特征相关性热力图 (Pearson)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()