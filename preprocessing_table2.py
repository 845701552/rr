import pandas as pd
from scipy.stats import ttest_rel,ttest_ind, levene,mannwhitneyu
from sklearn.feature_selection import SelectKBest
from sklearn import feature_selection
from sklearn.feature_selection import chi2
from scipy.stats.mstats import kruskalwallis
import numpy as np
from statsmodels.formula.api import ols
from scipy.stats import kstest,ks_2samp,shapiro,f_oneway,chi2_contingency
from sklearn.preprocessing import RobustScaler
data = pd.read_excel("C:/Users/gang.yu/Desktop/新变量数据V2.xlsx", sheet_name="总数据")
data_describe = pd.read_excel("C:/Users/gang.yu/Desktop/新变量数据V2.xlsx", sheet_name="总数据描述")
target = "PSP三分类"

#data.fillna(-1,inplace=True)
def out_put_statistics_info(data, data_describe):
    cols = list(data_describe["变量名称"].values)
    for i in cols:
        print(i)
    cols.remove("患者编号")
    cols.remove("PSP总评分")
    cols.remove("PSP三分类")
    cols.remove("PSP二分类（0）")
    cols.remove("PSP二分类（1）")
    cols.remove("No")
    dict_data = {}
    target_lable=list(set(data[target]))
    if len(set(data[target]))==3:
        for col in cols:
            type = data_describe[data_describe["变量名称"] == col]["变量类型"].values[0]
            vals = [data[col].notnull().sum()]
            if type == "数值(N)":
                "先判断是否方差齐次，才能使用独立t检验" \
                "样本大于5000：Kolmogorov-Smirnov tes" \
                "样本小于5000：shapiro-wilk" \

                if data.shape[0]>5000:
                    ks,p=kstest(data[data[col].notnull()][col], 'norm')
                else:
                    s,p=shapiro(data[data[col].notnull()][col])
                    if p> 0.05:
                        "anova"
                        "t检验 ,不能用 ,填哪一个t p"
                        df = data[[col,target]]
                        df=df[df[col].notnull()]
                        st,p=f_oneway(df[df[target] ==0][col].values,df[df[target] ==1][col].values,
                                      df[df[target] ==2][col].values)

                        vals.append(st)
                        vals.append("anova")
                        vals.append("anova")
                        vals.append(p)

                    else:
                        "三分类：kruskal-wallis"
                        df = data[[col,target]]
                        df=df[df[col].notnull()]
                        h, p = kruskalwallis(df[df[target] ==0][col].values,df[df[target] ==1][col].values,
                                      df[df[target] ==2][col].values)
                        vals.append(h)
                        vals.append("kruskal")
                        vals.append("kruskal")
                        vals.append(p)
                        "可能包含空值"
                    vals.append("%.2f±%.2f" % (data[data[target] == 0][col].mean(), data[data[target] == 0][col].std()))
                    vals.append("%.2f±%.2f" % (data[data[target] == 1][col].mean(), data[data[target] == 1][col].std()))
                    vals.append("%.2f±%.2f" % (data[data[target] == 2][col].mean(), data[data[target] == 2][col].std()))
            if type == "字符串":
                "************************"
                "卡方检验"
                cols_kf = [col,target]
                data_kf = data[data[col].notnull()][cols_kf]
                tt = data_kf.groupby([col, target])[target].count().unstack()
                tt.fillna(0, inplace=True)
                tt = np.array(tt)
                g, p, dof, expctd= chi2_contingency(tt)
                vals.append("卡方")
                vals.append("卡方")
                vals.append(g)
                vals.append(p)
                vals.append(np.nan)
                vals.append(np.nan)
                vals.append(np.nan)
            dict_data[col] = vals
    "**************************************************"
    # if len(set(data[target]))==2:
    #     for col in cols:
    #         type =data_describe[data_describe["变量名称"]==col]["变量类型"].values[0]
    #         vals =[data[col].notnull().sum()]
    #         if type=="数值(N)":
    #             "先判断是否方差齐次，才能使用独立t检验"
    #             if data.shape[0] > 5000:
    #                 ks, p = kstest(data[col], 'norm')
    #
    #             else:
    #                 s, p = shapiro(data[col])
    #                 if p>0.05:
    #                     "t检验 ,不能用 ,填哪一个t p"
    #                     t,p= ttest_ind(data[data[target]==0][col].values,data[data[target]==1][col].values
    #                                  ,equal_var=True)
    #                     vals.append(t)
    #                     vals.append(np.nan)
    #                     vals.append(np.nan)
    #                     vals.append(p)
    #
    #                 else:
    #                     "二分类：Mann-Whitney U test"
    #
    #                     m, p =mannwhitneyu(data[data[target] == 0][col].values, data[data[target] == 1][col].values,
    #                                          )
    #                     vals.append(m)
    #                     vals.append(np.nan)
    #                     vals.append(np.nan)
    #                     vals.append(p)
    #                     "可能包含空值"
    #                 vals.append("%.2f±%.2f" % (data[data[target] == 0][col].mean(), data[data[target] == 0][col].std()))
    #                 vals.append("%.2f±%.2f" % (data[data[target] == 1][col].mean(), data[data[target] == 1][col].std()))
    #                 vals.append("%.2f±%.2f" % (data[data[target] == 2][col].mean(), data[data[target] == 2][col].std()))
    #         if type == "字符串":
    #
    #
    #             "************************"
    #             #data[col].replace(-1,1000000, inplace=True)
    #
    #             "**********计算卡方值没有影响？**************"
    #             temp = feature_selection.SelectKBest(chi2, k=1). \
    #                 fit(np.array(data[col]).reshape(-1, 1),
    #                     np.array(data[target]).reshape(-1, 1))
    #             vals.append(np.nan)
    #             vals.append(np.nan)
    #             vals.append(temp.scores_.tolist()[0])
    #             vals.append(temp.pvalues_.tolist()[0])
    #             vals.append(np.nan)
    #             vals.append(np.nan)
    #             vals.append(np.nan)
    #         dict_data[col] = vals


    "**************************************************"
    indexs = ["样本数", "t值", "F值", "卡方值", "p"]
    for i in range(len(set(data[target]))):
        indexs.append("结局-{0}".format(i))
    total_describe_df = pd.DataFrame(dict_data, index=indexs)
    column = total_describe_df.columns
    total_describe_df.replace("null", " ", inplace=True)
    total_describe_np = np.array(total_describe_df).T
    total_describe_df = pd.DataFrame(total_describe_np, index=column, columns=indexs)
    print(total_describe_df)
    total_describe_df.to_csv("C:/Users/gang.yu/Desktop/jieju_group_describe_df_new.csv", encoding="utf_8_sig")
#out_put_statistics_info(data,data_describe)

# df=data[['LDL','PSP三分类']]

"全自动"
def str_group_precent(data, data_describe):
    cols = list(data_describe["变量名称"].values)
    cols.remove("患者编号")
    cols.remove("PSP总评分")
    cols.remove("PSP三分类")
    cols.remove("PSP二分类（0）")
    cols.remove("PSP二分类（1）")
    cols.remove("No")
    values_percent = []
    for col in cols:
        type = data_describe[data_describe["变量名称"] == col]["变量类型"].values[0]
        if type == "字符串":
            for col_kind in set(data[col]):
                col_kind_precent = []
                col_kind_precent.append(col + "_" + str(col_kind))
                for i in range(len(set(data[target]))):
                    col_kind_precent.append("%d(%.2f)" %
                                            (data[((data[col] == col_kind) & (data[target] == i))].shape[0],
                                             data[((data[col] == col_kind) & (data[target] == i))].shape[0] /
                                             data[data[target] == i].shape[0]))
                values_percent.append(col_kind_precent)
    cols = [" "]
    for i in range(len(set(data[target]))):
        cols.append(target + "_" + str(i))
    data_kind_percent = pd.DataFrame(values_percent, columns=cols)
    data_kind_percent.to_csv("C:/Users/gang.yu/Desktop/data_kind_percent_new.csv",encoding="utf_8_sig",index=False)
#str_group_precent(data, data_describe)


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
    total_describe_df.to_csv("C:/Users/gang.yu/Desktop/total_describe_df_new.csv",encoding="utf_8_sig")
out_put_total_describe(data,data_describe)