import pandas as pd
import numpy as np
data=pd.read_excel("D:/auto_preprocessing/分析数据输入模板-提供文件.xlsx",sheet_name="原始数据")
data_describe=pd.read_excel("D:/auto_preprocessing/分析数据输入模板-提供文件.xlsx",sheet_name="变量说明")

"变量类型只包含：字符串，数据(N),  结局没有包含"
def out_put_total_describe(data,data_describe):
    cols=list(data_describe["变量名称"].values)
    cols.remove("患者编号")
    cols.remove("No")
    dict1={}
    for col in cols:
        leixing=data_describe[data_describe["变量名称"] == col]["变量类型"].values[0]
        if leixing=="数值(N)":
            "注：取平均不能包含空的值"
            vals=[data_describe[data_describe["变量名称"]==col]["变量类型"].values[0],
                  data[col].notnull().sum(),
                  "%.1f%%" % ((data[col].isnull().sum()/data[col].isnull().count()) * 100),
                  round(data[data[col].notnull()][col].min(),2),
                  round(data[data[col].notnull()][col].max(),2),
                  round(data[data[col].notnull()][col].mean(),2),
                  round(data[data[col].notnull()][col].std(),2),
                  round(data[data[col].notnull()][col].quantile(0.75)-data[data[col].notnull()][col].quantile(0.25),2)
                  ]
            dict1[col] = vals
        if leixing=="字符串":
            vals = [data_describe[data_describe["变量名称"] == col]["变量类型"].values[0],
                    data[col].notnull().sum(),
                    "%.1f%%" % ((data[col].isnull().sum() / data[col].isnull().count()) * 100),
                    "null","null","null","null","null"
                    ]
            dict1[col] = vals
    indexs=["变量类型","样本数","缺失率","最小值","最大值","均值","标准差","四分位间距"]
    total_describe_df=pd.DataFrame(dict1,index=indexs)
    column=total_describe_df.columns
    total_describe_df.replace("null"," ",inplace=True)
    total_describe_np=np.array(total_describe_df).T
    total_describe_df = pd.DataFrame(total_describe_np, index=column,columns=indexs)
    print(total_describe_np)
    total_describe_df.to_csv("C:/Users/gang.yu/Desktop/total_describe_df.csv",encoding="utf_8_sig")
#out_put_total_describe(data,data_describe)