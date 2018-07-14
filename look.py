"参考博客：https://blog.csdn.net/iam_emily/article/details/79307373"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv("E:/pycharm project/compitition/train.csv")
#train=pd.read_csv("C:/Users/yu/Desktop/train_final.csv")

y=""
"1：分析目标变量"
def regression_target(train):
    print(train["SalePrice"].describe())
    sns.distplot(train['SalePrice'])
    plt.show()
    print("Skewness: %f" % train['SalePrice'].skew())
    print("Kurtosis: %f" % train['SalePrice'].kurt())
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train.shape[0]), np.sort(train['SalePrice'].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.show()
#regression_target(train)

def class_target(train):
    print(train["SalePrice"].describe())
    plt.bar(train[col].value_counts().index, train[col].value_counts())
    plt.show()
#class_target(train)

"2:变量的unique取值个数"
def Number_of_unique_feat(train):
    cols = train.columns
    uniques = [len(train[col].unique()) for col in cols]
    sns.set(font_scale=1.1)
    ax = sns.barplot(cols, uniques, palette='hls', log=True)
    ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique per feature')
    for p, uniq in zip(ax.patches, uniques):
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 10,
                uniq,
                ha="center")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()
#Number_of_unique_feat(train)

"3:缺失值可视化"
def look_mission(train):
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)
    sns.boxplot(missing_data.index,missing_data["percent"])
    plt.show()
#look_mission(train)
"热力图"
def hm(train):
    corrmat = train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    "可以用博客看图片带下拉框"
    plt.savefig("C:/Users/yu/Desktop/heatmap.png")
    plt.show()

    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.savefig("C:/Users/yu/Desktop/class_category_cnt_ratio.png")
    plt.show()
#hm(train)

"4：得到数字，类目变量，进行下一步分析画图"
def get_num_object(train):
    num_features= [f for f in train.columns if train.dtypes[f] != 'object']#数值特征
    # pop the target-Saleprice
    num_features.pop()
    print('numerical feature length:',len(num_f))
    category_features = [f for f in train.columns if train.dtypes[f] == 'object']#训练集类别特征
    print('category feature length:',len(category_features))
    return num_features,category_features
#num_features,category_features=get_num_object(train)
"5:"
def class_category_cnt_ratio(train):
    "类目变量分布取值个数"
    col="sms_hh_min"
    plt.figure(figsize=(8,4))
    plt.colors()
    p = sns.color_palette()
    plt.subplot(121)
    plt.title(col+"cnt")
    plt.bar(train[col].value_counts().index, train[col].value_counts(),color=p[np.random.randint(6)])
    "每个取值的风险率"
    plt.subplot(122)
    plt.title(col+"ratio")


    data=train.groupby(col,as_index=False)["label"].agg({"sum1":"sum","size":"count"})
    data["risk_ratio"]=data["sum1"]/data["size"]
    plt.bar(data.iloc[:, 0], data.iloc[:, 3],color=p[np.random.randint(6)])
    "可以用博客看图片带下拉框"
    plt.savefig("C:/Users/yu/Desktop/class_category_cnt_ratio.png")
    plt.show()

    "分析在in_out情况下的风险率"
    # data=train[train["in_out"]==0].groupby(col,as_index=False)["label"].agg({"sum1":"sum","size":"count"})
    # data["riskv"]=data["sum1"]/data["size"]
    # print(data.sort_values(by="riskv",ascending=False))
    # plt.title(col+"ratio")
    # plt.bar(data.iloc[:,0],data.iloc[:,3])
    # plt.show()
    #
    # data=train[train["in_out"]==1].groupby(col,as_index=False)["label"].agg({"sum1":"sum","size":"count"})
    # data["riskv"]=data["sum1"]/data["size"]
    # print(data.sort_values(by="riskv",ascending=False))
    # plt.title("neg_hh_ratio")
    # plt.bar(data.iloc[:,0],data.iloc[:,3])
    # plt.show()
#class_category_cnt_ratio(train)

def regression_category_cnt_target_mean(data):
    """类别取值的数量，对应应变量的均值,可以增加for循环多个变量"""
    p = sns.color_palette()
    plt.subplot(121)
    plt.title("v_hh_ratio")
    plt.bar(train["MSZoning"].value_counts().index, train["MSZoning"].value_counts(),color=p[np.random.randint(6)])
    data = train.groupby("MSZoning", as_index=False)["SalePrice"].agg({"mean": "mean"})
    plt.subplot(122)
    plt.title("pos_hh_ratio")
    plt.bar(data.iloc[:, 0], data.iloc[:, 1])
    "可以用博客看图片带下拉框"
    plt.savefig("C:/Users/yu/Desktop/regression_category_cnt_target_mean.png")
    plt.show()
#regression_category_cnt_target_mean(train)


"4:category类别中每个取值，对应应变量的boxplot"
def regression_category_target_boxplot(train):
    """category类别中每个取值，对应应变量的boxplot"""
    #value_vars=[]
    #id_vars=
    f = pd.melt(train, id_vars=['SalePrice'], value_vars=["MSZoning","LotConfig"])
    g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size=5)
    g.map(sns.boxplot, "value", "SalePrice")
    plt.show()
#regression_category_target_boxplot(train)


"5:结合图4分析"
"回归：类别特征取值，应变量变换情况,判断各因素(变量)对试验指标的影响是否显著"
"""单因素方差分析适用于三组以上平均数差异的检验
要求数据满足正态性、独立性和 差齐性的要求，
对数据要求较高，数据为连续数据。
卡方检验要求则没那么高。     
例子中如果要分析不同岗位之间的工资差异，
在数据较多满足正态、方差齐性、分析的数据超过三组条件
下应该使用单因素方差分析；
而婚姻状况的数据分为已婚、未婚、离婚情况，
不是连续型数据，所以采用卡方检验。
"""
"""
def anova(frame):
    import scipy.stats as statsMM
    anv = pd.DataFrame()
    anv['feature'] = category_features
    pvals = []
    for c in category_features:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = statsMM.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')
a = anova(train)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
plt.show()
"""


"6:regession:类别特征与目标变量boxplot"
"""
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=category_f)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False,sharey=False,size=5)
g = g.map(boxplot, "value", "SalePrice")
plt.savefig("C:/Users/yu/Desktop/many.png")
save(obj=column(g), filename=path, title="outputTest")
plt.show()
"""



"7:回归：数字特征与目标的regplot"
"""
def jointplot(x,y,**kwargs):
    try:
        sns.regplot(x=x,y=y)
    except Exception:
        print(x.value_counts())
f = pd.melt(train, id_vars=['SalePrice'], value_vars=num_f)
g = sns.FacetGrid(f,col='variable',col_wrap=3,sharex=False,sharey=False,size=5)
g = g.map(sns.regplot,'value','SalePrice')
"可以用博客看图片带下拉框"
plt.savefig("C:/Users/yu/Desktop/regression_category_cnt_target_mean.png")
plt.show()
"""


"8:整体相关性"
#特征互相关分析与选取相关分析与选取
num_features = [f for f in train.columns if train.dtypes[f] != 'object']
def spearman(frame, features):
    '''
    采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
    此相关系数简单来说，可以对上述数值化处理后的等级变量及其它与房价的相关性进行更好的评价
    （特别是对于非线性关系）
    '''
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
    "可以用博客看图片带下拉框"
    plt.savefig("C:/Users/yu/Desktop/spearman_num.png")
    plt.show()
"将类别编码为数字后一起分析，features = num_data + category_encoded"
#spearman(train,num_features)


"9：编码"
"""
def encode(frame, feature):
    '''
    对所有类型变量，依照各个类型变量的不同取值对应的样本集内房价的均值，按照房价均值高低
    对此变量的当前取值确定其相对数值1,2,3,4等等，
    同理class中类目值可以按照风险率的高低依次编码
    相当于对类型变量赋值使其成为连续变量。
    注意：此函数会直接在原frame的DataFrame内创建新的一列来存放feature编码后的值。
    '''
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['price_mean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    # 上述 groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    for attr_v, score in ordering.items():
        # e.g. qualitative[2]: {'Grvl': 1, 'MISSING': 3, 'Pave': 2}
        frame.loc[frame[feature] == attr_v, feature+'_E'] = score

category_encoded = []
# 由于category集合中包含了非数值型变量和伪数值型变量（多为评分、等级等，其取值为1,2,3,4等等）两类
# 因此只需要对非数值型变量进行encode()处理。
# 如果采用One-Hot编码，则整个qualitative的特征都要进行pd.get_dummies()处理
category_left = ['BldgType','Condition1','Condition2','Electrical','Exterior1st',
                'Exterior2nd','Foundation','Functional','Heating','HouseStyle',
                'LandContour','LandSlope','LotConfig','LotShape','MSZoning',
                'PavedDrive','RoofMatl','RoofStyle','SaleCondition']
for q in category_left:
    encode(all_data,q)
    category_encoded.append(q+'_encode')
all_data.drop(category_left, axis=1, inplace=True) 
# 离散变量已经有了编码后的新变量，因此删去原变量
print(category_encoded)
print(len(category_encoded))
"""

"10:聚合,单个聚合（可以），多个聚合"
def agg(train):
    agg_cols=[]
    agg=""
    for col in agg_cols:
        agg_temp=train.groupby("uid",as_index=False)[col].agg({col+"_max":"max",col+"_min":"min",})
        agg=pd.merge(agg,agg_temp,how="left",on="uid")
    return agg
def sigle_cate_regression():
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000);
def sigle_cate():
    data = train.groupby("OverallQual")["SalePrice"].agg({"mean": np.mean, "max": np.max})
    data.plot.bar(stacked=True)
    #data.plot(stacked=True)
    plt.show()
