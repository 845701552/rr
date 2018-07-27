import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy import interp
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc,roc_curve,classification_report,average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split,KFold,StratifiedKFold
from sklearn.linear_model import LassoCV,RidgeCV,LinearRegression,ElasticNet,ElasticNetCV,Lasso,SGDClassifier,LogisticRegression
import numpy as np
import seaborn as sns


def random_grid():
    column=["新生儿是否感染",
            "产程开始_胎膜早破时间差_h",
            "母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
            "孕周_足月为0_近足月早产为1_远足月早产为2",
            "白细胞计数",
            "中性粒细胞比例",
            "关联CRP",
            "B族链球菌",
            "羊水情况_清为0_混为1_胎粪为2"]
    data=pd.read_excel(r"C:\Users\gang.yu\Desktop\10291病人.xlsx")[column]
    data.fillna(-1,inplace=True)
    #data.to_csv(r"C:\Users\gang.yu\Desktop\10291病人_get_col.csv",encoding="utf_8_sig")
    column_str=["母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
                "孕周_足月为0_近足月早产为1_远足月早产为2",
                "关联CRP",
                "B族链球菌",
                "羊水情况_清为0_混为1_胎粪为2"
                ]
    for col in column_str:
            data[col]=data[col].astype(str)
    data=pd.get_dummies(data,columns=column_str,prefix=column_str,prefix_sep="+")

    print(data.isnull().sum())
    print(data.head())



def get_data():
    column = ["新生儿是否感染",
              "产程开始_胎膜早破时间差_h",
              "母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
              "孕周_足月为0_近足月早产为1_远足月早产为2",
              "白细胞计数",
              "中性粒细胞比例",
              "关联CRP",
              "B族链球菌",
              "羊水情况_清为0_混为1_胎粪为2"]
    data = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291病人.xlsx",na_values=[])
    beizhu = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291病人.xlsx",sheet_name="备注")
    data.fillna(-1,inplace=True)
    str_col=list(beizhu[beizhu["变量类型"]=="分类变量"]["变量原始"].values)
    str_col.remove("新生儿是否感染")
    print("分类变量数量: "+str(len(str_col)))
    for col in str_col:
        data[col] = data[col].astype(int)
        data[col]=data[col].astype(str)
    data = pd.get_dummies(data, columns=str_col, prefix=str_col, prefix_sep="+")
    onehot_cols=data.columns
    need_col = []
    for col in onehot_cols:
        if "+0" not in col:
            need_col.append(col)
    need_col.remove("id")
    need_col.remove("12hK")
    train=data[need_col]
    print(train.info())
    print(train.shape)
    return  train


# train=get_data()
# y=train["新生儿是否感染"]
# X=train.drop("新生儿是否感染",axis=1)

def get_data8():
    column = ["新生儿是否感染",
              "产程开始_胎膜早破时间差_h",
              "母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
              "孕周_足月为0_近足月早产为1_远足月早产为2",
              "白细胞计数",
              "中性粒细胞比例",
              "关联CRP",
              "B族链球菌",
              "羊水情况_清为0_混为1_胎粪为2",
        ]
    data = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291new.xlsx",na_values=[])[column]
    validation = pd.read_excel(r"C:\Users\gang.yu\Desktop\validation分析数据.xlsx")[column]


    train_len=data.shape[0]
    data_all=pd.concat([data,validation])

    #print(data_crp.shape)
    data.fillna(-1,inplace=True)
    data_all.fillna(-1, inplace=True)
    column_str = ["母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
                  "孕周_足月为0_近足月早产为1_远足月早产为2",
                  "关联CRP",
                  "B族链球菌",
                  "羊水情况_清为0_混为1_胎粪为2",

                  ]
    for col in column_str:
        data[col] = data[col].astype(int)
        data[col] = data[col].astype(str)
    data = pd.get_dummies(data, columns=column_str, prefix=column_str, prefix_sep="+")
    onehot_cols=data.columns

    need_col = []

    for col in onehot_cols:
        if "+0" not in col:
            need_col.append(col)
    train=data[need_col]
    print(train.info())
    print(train.shape)
    return  train

# train8=get_data8()
# yy=train8["新生儿是否感染"]
# XX=train8.drop("新生儿是否感染",axis=1)



def look():
    col="关联CRP"
    column = ["新生儿是否感染",
              "产程开始_胎膜早破时间差_h",
              "母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
              "孕周_足月为0_近足月早产为1_远足月早产为2",
              "白细胞计数",
              "中性粒细胞比例",
              "关联CRP",
              "B族链球菌",
              "羊水情况_清为0_混为1_胎粪为2"]
    data = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291病人.xlsx", na_values=[])[column]
    data.fillna(-1,inplace=True)
    validation = pd.read_excel(r"C:\Users\gang.yu\Desktop\validation分析数据.xlsx")[column]

    # sns.kdeplot(data[data["新生儿是否感染"]==1][col],label="train1")
    # sns.kdeplot(data[data["新生儿是否感染"] ==0][col], label="train0")
    # sns.kdeplot(validation[validation["新生儿是否感染"] == 1][col],label="val1")
    # sns.kdeplot(validation[validation["新生儿是否感染"] == 0][col], label="val0")
    # plt.show()


    plt.hist(data[data["新生儿是否感染"]==1][col],label="train1")
    plt.hist(data[data["新生儿是否感染"] == 0][col], label="train0")
    plt.figure()
    plt.hist(validation[validation["新生儿是否感染"]==1][col],label="val1")
    plt.hist(validation[validation["新生儿是否感染"] == 0][col], label="val0")
    plt.show()
#look()






def get_data8_val734():
    column = ["新生儿是否感染",
              "产程开始_胎膜早破时间差_h",
              "母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
              "孕周_足月为0_近足月早产为1_远足月早产为2",
              "白细胞计数",
              "中性粒细胞比例",
              "关联CRP",
              "B族链球菌",
              "羊水情况_清为0_混为1_胎粪为2",
              ]
    data = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291病人(3).xlsx", na_values=[])[column]
    column_id=column
    column_id.append("id")
    pred734 = pd.read_excel(r"C:\Users\gang.yu\Desktop\734患者.xlsx")[column_id]

    train_len = data.shape[0]
    id734=pred734["id"]
    pred734.drop("id",inplace=True,axis=1)

    data_all = pd.concat([data,pred734])
    data_all.fillna(-1, inplace=True)
    column_str = ["母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为",
                  "孕周_足月为0_近足月早产为1_远足月早产为2",
                  "关联CRP",
                  "B族链球菌",
                  "羊水情况_清为0_混为1_胎粪为2",
                  ]
    for col in column_str:
        data_all[col] = data_all[col].astype(int)
        data_all[col] = data_all[col].astype(str)
    data_all = pd.get_dummies(data_all, columns=column_str, prefix=column_str, prefix_sep="+")
    onehot_cols = data_all.columns
    need_col = []
    for col in onehot_cols:
        if "+0" not in col:
            need_col.append(col)
    train = data_all[need_col][:train_len]
    pred734=data_all[need_col][train_len:]
    return train,pred734,id734
#train_pred734,pred734,id734=get_data8_val734()



def logist_grid(train_val):
    y = train_val["新生儿是否感染"]
    XX = train_val.drop("新生儿是否感染",axis=1)
    std=StandardScaler()
    X=std.fit_transform(XX)
    params = {"C": [ 0.01,0.1,1,10,50,100],
              "penalty": ["l1", "l2"],
             # "class_weight": [{0: 1, 1: 4}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 7}]
             },
    clf = LogisticRegression(C=10,random_state=2018)



    #param_test2 = {'C': [0.7, 0.8, 0.9, 1.0]}


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    gsearch1 = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc',
                            cv=cv, verbose=1)
    gsearch1.fit(X, y)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    print(gsearch1.cv_results_)
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=2017)
    metric = np.mean((cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))
    print("cross_val_score :"+str(metric))


def sgd_gird(X,y):
    """
    l2:
    {'alpha': 0.1, 'class_weight': {0: 1, 1: 5}, 'l1_ratio': 0.9, 'loss': 'log', 'penalty': 'l2'}
    roc: 0.8888928975106992
    {'alpha': 0.1, 'class_weight': {0: 1, 1: 5}, 'l1_ratio': 0.01, 'loss': 'hinge', 'penalty': 'elasticnet'}
     roc:0.8852947124329595


     {'alpha': 0.1, 'class_weight': {0: 1, 1: 6}, 'l1_ratio': 0.5, 'loss': 'hinge', 'penalty': 'elasticnet'}
     f1:0.3606466901929592
     [0.87186536 0.81624025 0.76779623 0.79198332 0.81221686]
     test_mean_score:0.8120204027815813

    """
    params={"alpha":[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
            "l1_ratio":[.01, .1, .5, .9, .99],
            "penalty":["elasticnet","l1","l2"],
            "class_weight":[{0: 1, 1:4},{0: 1, 1:3},{0: 1, 1:5},{0: 1, 1:6},{0: 1, 1:7}],
            "loss":["hinge","log"],
            }
    sgd = SGDClassifier()#
    cv = StratifiedKFold(n_splits=5,random_state=2018)
    grid=GridSearchCV(sgd,param_grid=params,scoring="f1",cv=cv,n_jobs=-1,verbose=4)
    grid.fit(X,y)
    print(grid.best_params_)
    print("f1:"+str(grid.best_score_))



def grid_rf(train_val):
    """


    """
    y= train_val["新生儿是否感染"]
    X = train_val.drop("新生儿是否感染",axis=1)
    params={ #'n_estimators':[100,300,500,800,1000,1200,1500],
            #'max_depth': range(2, 10, 1),
          #'max_features': range(2, 8, 1),
         #"class_weight":[{0: 1, 1:4},{0: 1, 1:5},{0: 1, 1:6},{0: 1, 1:8},{0: 1, 1:10},{0: 1, 1:50},{0: 1, 1:20},{0: 1, 1:30},{0: 1, 1:40}],
          'min_samples_split': [2,3,5,6,10],
          'min_samples_leaf': [1,2,4,6,8],
            }
    rf=RandomForestClassifier(
        n_estimators= 300,
        # max_depth=7,
        # max_features=4,
        # class_weight={0: 1,1:5},
        # min_samples_split=5,
        # min_samples_leaf=2,
        random_state=2018,
        )

    cv = StratifiedKFold(n_splits=5, random_state=2018)
    grid=GridSearchCV(rf,param_grid=params,scoring="roc_auc",cv=cv,n_jobs=-1,verbose=4)
    grid.fit(X,y)
    print(grid.best_params_)
    print("auc_roc:"+str(grid.best_score_))





sgd_f1=SGDClassifier(loss="hinge", penalty='elasticnet', alpha=0.1,l1_ratio=0.5, class_weight={0: 1, 1:6})

sgd_train=SGDClassifier(loss="log", penalty='l2', alpha=0.1,l1_ratio=0.9, class_weight={0: 1, 1:5})
rf=RandomForestClassifier(
    n_estimators=1000,
    max_depth=7,
    max_features=4,
    class_weight={0: 1,1: 5},
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=2018,
)



def cross_val_score1(model,X,y):
    cv = StratifiedKFold(n_splits=5, random_state=2018)
    cv_scores=cross_val_score(model,X,y,cv=cv,scoring="roc_auc")
    print(cv_scores)
    mean_score1=np.mean(cv_scores)
    print("test_mean_score:"+str(mean_score1))



def  sgd_predict(X,y,sgd_train):
    sgd_train.fit(X, y)
    feat_importances=sgd_train.coef_
    feat_importances=list(np.hstack(feat_importances.flat))
    print(len(feat_importances))
    # feat_importances = pd.DataFrame({"var":column, "important": feat_importances})
    # feat_importances.to_csv("C:/Users/gang.yu/Desktop/gfy_sgd_feat_importances1.csv", encoding="utf_8_sig")

def crp_rf(train_val):
    train_crp=train_val[train_val["关联CRP+-1"]==0]
    print("crp==1")
    print(train_crp.shape)
    y_crp=train_crp["新生儿是否感染"]
    X_crp=train_crp.drop("新生儿是否感染",axis=1)
    y=train_val["新生儿是否感染"]
    X =train_val.drop("新生儿是否感染", axis=1)
    rf = RandomForestClassifier(
        n_estimators= 800,
        max_depth=5,
        class_weight={0: 1,1:10},
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=10,
    )
    rf.fit(X,y)
    crp=rf.predict_proba(X_crp)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_crp,crp)
    fpr, tpr, thresholds_roc = roc_curve(y_crp,crp)
    top15_all_thresholds = pd.DataFrame({"thresholds": thresholds})
    top15_all_pr=pd.DataFrame({"precision":precision,"recall":recall})
    top15_all_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr,"thresholds":thresholds_roc})
    "2个坐标文件"
    top15_all_roc.to_csv("C:/Users/gang.yu/Desktop/10291_roc1.csv", index=False)
    top15_all_thresholds.to_csv("C:/Users/gang.yu/Desktop/10291_threshold.csv", index=False)
    top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/10291_pr1.csv", index=False)

    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Reference Line', alpha=.8)
    plt.plot(fpr, tpr,color='b', lw=2, alpha=0.8,
             label=' (AUC = %0.3f)' % ( roc_auc))
    roc_auc = auc(recall, precision)
    plt.legend(loc="upper right")
    plt.figure()
    plt.plot(recall,precision, lw=2, alpha=0.8,
             label=' (AUC = %0.2f)' % ( roc_auc))
    plt.legend(loc="upper right")
    print()
    plt.show()


def k_fold_roc_pr(train_val):
    y = train_val["新生儿是否感染"]
    X = train_val.drop("新生儿是否感染", axis=1)
    std=StandardScaler()
    X=std.fit_transform(X)
    X=np.array(X)

    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        max_features=4,
        class_weight={0: 1, 1: 5},
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=2018,
    )
    lr = LogisticRegression(C=1, random_state=2018)
    pres = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    i = 0


    all_predict_proba=pd.DataFrame({"probas":[]})
    all_predict_label = pd.DataFrame({"label": []})
    for train, test in cv.split(X, y):
        probas_ =rf.fit(X[train], y[train]).predict_proba(X[test])
        temp_label=pd.DataFrame({"label": y[test]})
        temp = pd.DataFrame({"probas": probas_[:, 1]})
        all_predict_proba=pd.concat([all_predict_proba,temp])
        all_predict_label=pd.concat([all_predict_label,temp_label])
        precision, recall,thresholds = precision_recall_curve(y[test], probas_[:, 1])
        pres.append(interp(mean_recall[::-1],recall[::-1], precision[::-1]))
        pres[-1][0] = 0.0
        pr_auc = average_precision_score(y[test], probas_[:, 1])
        #roc_auc = auc(recall,precision)
        aucs.append(pr_auc)
        plt.plot(recall,precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.2f)' % (i, pr_auc))
        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Luck', alpha=.8)
    print("")
    print(aucs)



    all_predict_proba.to_csv("C:/Users/gang.yu/Desktop/proba10291_rf.csv", index=False)
    #proba=pd.concat([all_predict_label,all_predict_proba],axis=1)
    #proba.to_csv("C:/Users/gang.yu/Desktop/proba10291_lr.csv", index=False)

    precision, recall, thresholds = precision_recall_curve(all_predict_label["label"],all_predict_proba["probas"])

    fpr, tpr, thresholds_roc = roc_curve(all_predict_label["label"],all_predict_proba["probas"])
    top15_all_thresholds = pd.DataFrame({"thresholds": thresholds})




    top15_all_pr=pd.DataFrame({"precision":precision,"recall":recall})
    "*********************************"
    " pr_加入第一行，从0-1"
    # print("**")
    # print(y.sum()/y.count())
    # top15_all_pr_add = pd.DataFrame({"precision": [y.sum()/y.count()], "recall": [1]})
    # top15_all_pr=pd.concat([top15_all_pr_add,top15_all_pr])
    #
    # pr_auc=average_precision_score(y[test],probas_[:,1])
    # plt.figure()
    # plt.plot(top15_all_pr["recall"], top15_all_pr["precision"], lw=2, alpha=0.8,
    #          label=' (AUC = %0.2f)' % (pr_auc))
    # plt.legend(loc="upper right")
    # plt.xlabel('Recall1,%')
    # plt.ylabel('Precision1,%')
    # plt.show()
    "*********************************"



    top15_all_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr,"thresholds":thresholds_roc})
    "2个坐标文件"
    top15_all_roc.to_csv("C:/Users/gang.yu/Desktop/top8_roc1.csv", index=False)
    top15_all_thresholds.to_csv("C:/Users/gang.yu/Desktop/top8_threshold.csv", index=False)
    #top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/top8_pr1.csv", index=False)


    # all_predict_label.to_csv("C:/Users/gang.yu/Desktop/model1_top15_label.csv", index=False)
    # all_predict_proba.to_csv("C:/Users/gang.yu/Desktop/model1_top15_predict_proba.csv", index=False)
    #print(all_predict_proba)


    print("mean_auc:"+str(np.mean(aucs)))
    mean_pre = np.mean(pres, axis=0)
    mean_pre[-1] = 1.0


    mean_auc=average_precision_score(y[test],probas_[:,1])

    #mean_auc = auc(mean_recall, mean_pre)



    std_auc = np.std(aucs)

    interval = stats.t.interval(0.95, 4, np.mean(aucs),std_auc)

    # model1_top15_roc_df=pd.DataFrame({"mean_recall":mean_recall,"mean_pre":mean_pre[::-1]})
    # model1_top15_roc_df.to_csv("C:/Users/gang.yu/Desktop/gfy_top15_pr.csv",index=False)
    plt.plot(mean_recall, mean_pre[::-1], color='b',
             label=r'Mean PR (AUC = %0.3f $\pm$ %0.2f))' % (mean_auc,std_auc),
             lw=2, alpha=.8)
    std_pre= np.std(pres, axis=0)
    pres_upper = np.minimum(mean_pre + std_pre, 1)
    pres_lower = np.maximum(mean_pre - std_pre, 0)
    # plt.fill_between(mean_recall, pres_lower[::-1], pres_upper[::-1], color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')  #  interval=(%0.2f,%0.2f)   % (interval[0],interval[1])


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall,%')
    plt.ylabel('Precision,%')
    #plt.title('precision recall')
    plt.legend(loc="upper right")
    plt.figure()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = lr.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Reference Line', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    print("")
    print(aucs)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    interval = stats.t.interval(0.95,4,np.mean(aucs) ,std_auc)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc,std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')#  interval=(%0.2f,%0.2f) % (interval[0],interval[1]))


    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('1-Specificity,%')
    plt.ylabel('Sencitivity,%')
    #plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()



"注：validation分析数据.xlsx名字改" \
"新的数据去除变量:母亲宫腔培养_有致病菌生长为1_无致病菌生长为0_没有检查为"
"*******读取train val数据：亚编码"
def get_data8_val():
    column = ["新生儿是否感染",
              "产程开始_胎膜早破时间差_h",
              "孕周_足月为0_近足月早产为1_远足月早产为2",
              "白细胞计数",
              "中性粒细胞比例",
              "关联CRP",
              "B族链球菌",
              "羊水情况_清为0_混为1_胎粪为2",
              ]
    data = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291new.xlsx", na_values=[])[column]
    validation = pd.read_excel(r"C:\Users\gang.yu\Desktop\validation数据整理V4.xlsx",sheet_name="1362分析数据")
    id=validation["住院号"]
    validation=validation[column]
    train_len = data.shape[0]
    data_all = pd.concat([data, validation])


    data_all.fillna(-1, inplace=True)
    column_str = [
                  "孕周_足月为0_近足月早产为1_远足月早产为2",
                  "关联CRP",
                  "B族链球菌",
                  "羊水情况_清为0_混为1_胎粪为2",
                  ]
    for col in column_str:
        data_all[col] = data_all[col].astype(int)
        data_all[col] = data_all[col].astype(str)
    data_all = pd.get_dummies(data_all, columns=column_str, prefix=column_str, prefix_sep="+")
    onehot_cols = data_all.columns
    need_col = []
    for col in onehot_cols:
        if "+0" not in col:
            need_col.append(col)
    train = data_all[need_col][:train_len]
    test=data_all[need_col][train_len:]
    print("train:"+str(train.shape[0]))
    print("test:" + str(test.shape[0]))
    return train,test,id







def rf_val(train_val,test_val,id):
    "只预测关联crp"
    #test_val=test_val[test_val["关联CRP+-1"]==0]
    print("new")
    print(test_val.shape[0])
    yy = train_val["新生儿是否感染"]
    XX = train_val.drop("新生儿是否感染",axis=1)


    y_val=test_val["新生儿是否感染"]
    X_val= test_val.drop("新生儿是否感染", axis=1)
    column=X_val.columns

    std = StandardScaler()
    XX_std = std.fit_transform(XX)
    X_val_std = std.transform(X_val)

        # n_estimators= 800,
        # max_depth=5,
        # class_weight={0: 1,1:10},
        # min_samples_split=5,
        # min_samples_leaf=2,
        # random_state=10,



    rf = RandomForestClassifier(
        n_estimators= 800,
        max_depth=5,
        class_weight={0: 1,1:10},
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=10,
    )

    lr = LogisticRegression(C=1,random_state=2018)
    lr.fit(XX_std,yy)

    rf.fit(XX,yy)
    print(len(column))
    #print(rf.feature_importances_)
    #top15_all_pr = pd.DataFrame({"column": column, "proba": rf.feature_importances_})
    #top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/rf_features_important.csv", index=False,encoding="utf_8_sig")


    crp = lr.predict_proba(X_val_std)[:,1]



    #crp_result=rf.predict(X_val)
    #print(classification_report(y_val,crp_result))
    top15_all_pr=pd.DataFrame({"id":id.values,"proba_lr":crp})
    #top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/proba_lr_RF.csv", index=False)
    #top15_all_pr=pd.DataFrame({"id":id.values,"proba":crp})
    #top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/validation_v4.csv",index=False)



    precision, recall, thresholds = precision_recall_curve(y_val, crp)
    print("******pr_auc")
    print(average_precision_score(y_val,crp))
    pr_auc = auc( recall,precision)
    print(pr_auc)
    "**********"



    fpr, tpr, thresholds_roc = roc_curve(y_val, crp)
    print("********roc")
    print(roc_auc_score(y_val, crp))
    print(auc(fpr, tpr))
    print("****")

    top15_all_thresholds = pd.DataFrame({"thresholds": thresholds})

    top15_all_pr=pd.DataFrame({"precision":precision,"recall":recall})
    top15_all_pr_add = pd.DataFrame({"precision": [y_val.sum()/y_val.count()], "recall": [1]})
    " pr_加入第一行，从0-1"
    top15_all_pr=pd.concat([top15_all_pr_add,top15_all_pr])
    top15_all_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr,"thresholds":thresholds_roc})
    "2个坐标文件"


    #top15_all_roc.to_csv("C:/Users/gang.yu/Desktop/roc_coordinate.csv", index=False)
    # top15_all_thresholds.to_csv("C:/Users/gang.yu/Desktop/pr_thresholds_coordinate.csv", index=False)
    #top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/pr1.csv", index=False)
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Reference Line', alpha=.8)
    plt.plot(fpr, tpr, color='b', lw=2, alpha=0.8,
             label=' (AUC = %0.3f)' % (roc_auc))
    "   "
    pr_auc=average_precision_score(y_val,crp)
    #pr_auc = auc(top15_all_pr["recall"],top15_all_pr["precision"])
    plt.xlabel('1-Specificity,%')
    plt.ylabel('Sencitivity,%')
    plt.legend(loc="lower right")
    plt.figure()
    plt.plot(top15_all_pr["recall"], top15_all_pr["precision"], lw=2, alpha=0.8,
             label=' (AUC = %0.2f)' % (pr_auc))
    plt.legend(loc="upper right")
    plt.xlabel('Recall,%')
    plt.ylabel('Precision,%')
    plt.show()



def rf_pred734(train_val,pred734,id734):
    #test_val=test_val[test_val["关联CRP+1"]==0]

    yy = train_val["新生儿是否感染"]
    XX = train_val.drop("新生儿是否感染",axis=1)
    pred_X= pred734.drop("新生儿是否感染", axis=1)


    rf = RandomForestClassifier(
        n_estimators= 800,
        max_depth=5,
        class_weight={0: 1,1:10},
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=10,
    )
    rf.fit(XX,yy)
    pred = rf.predict_proba(pred_X)[:,1]

    top15_all_pr=pd.DataFrame({"id":id734.values,"proba":pred})
    top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/pred374.csv",index=False)




if __name__ == '__main__':
    "1:修改get_data8_val()路径后，只打开rf_val，直接运行，"
    "2:只预测有crp的rf_val(train_val,test_val)中"
    "打开test_val=test_val[test_val[关联CRP+-1]==0]"
    train_val, test_val, id = get_data8_val()
    y = train_val["新生儿是否感染"]
    X = train_val.drop("新生儿是否感染", axis=1)




    #rf_val(train_val,test_val,id)
    #rf_validation_more_feature()
   # sgd_gird(train)
   #sgd_predict(train,sgd_train)
    #cross_val_score1(rf,X,y)
    grid_rf(train_val)
   #print("good!")
    #k_fold_roc_pr(train_val)
    #crp_rf(train_val)
    #logist_grid(train_val)

    #rf_pred734(train_val, pred734, id734)