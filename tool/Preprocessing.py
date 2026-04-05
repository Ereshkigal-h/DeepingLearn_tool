import pandas as pd
import numpy as np
def category_data(data_input,label_dict:dict,fill_value:dict=None,one_hot:list=None):
    """
        针对结构化数据的类别特征预处理流水线 (Label Encoding & One-Hot Encoding)
        参数 (Parameters):
        ------------------
        data_input : pd.DataFrame
            输入的原始数据表格。
        label_dict : dict, 默认为 None
            类别映射字典。格式为 {'列名': {'类别1': 映射值1, '类别2': 映射值2}}。
            用于具有明显大小或等级顺序的类别特征（如：学历、职称）。
            也可用于独热编码前的数据清洗（如：统一英文字母大小写）。
        fill_value : dict, 默认为 None
            缺失值填充策略字典。格式为 {'列名': 填充值}。
            如果不指定，默认会将 label_dict 中涉及的列的缺失值填充为 -1。
        one_hot : list, 默认为 None
            需要进行独热编码 (One-Hot) 的列名列表。
            用于没有明确大小顺序的类别特征（如：性别、城市）。

        返回 (Returns):
        ---------------
        pd.DataFrame
            经过编码处理后的全新数据表格。
        """
    if one_hot is None:
        one_hot=[]
    if fill_value is None:
        fill_value={col:-1 for col in label_dict.keys()}
    print(f"进行快速编码处理，原始数据维度: {data_input.shape}")
    data_category = data_input.copy()
    temp=0
    print(f"开始处理字典映射，检测到有{len(label_dict)}配置规则")
    for col,mapping in label_dict.items():
        if col in data_category.columns:
            if col not in one_hot:
                data_category[col]=data_category[col].map(mapping).fillna(fill_value.get(col,-1))
                temp += 1
    if one_hot:
        print(f"正在处理one_hot，目标列: {one_hot}")
        for col in one_hot:
            if col in data_category.columns:
                if col in label_dict.keys():
                    data_category[col]=data_category[col].map(label_dict[col]).fillna("unknown")
                else:
                    data_category[col]=data_category[col].fillna("unknown")
        data_category = pd.get_dummies(data_category, columns=one_hot, dtype=int)
    else:
        print(f"无独立编码，跳过处理")
    return data_category
