from tool import check_data,Preprocessing
import pandas as pd
import numpy as np
import seaborn as sns

if __name__=='__main__':
    pops,df=check_data.load_data("data/train.csv",pop=["id"])
    df_temp=df
    check_data.plot_features_frequency(df,df.columns,kde=False)
