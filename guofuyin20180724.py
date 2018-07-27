import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.svm import SVC,SVR
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import auc, roc_curve, classification_report, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import Lasso, SGDClassifier, LogisticRegression,ElasticNet,LinearRegression
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle


"得到roc_pr曲线：y_test,pre_test,pre_test_label"
from roc_pr import get_roc_pr
def preprocessing_for_sgd():
    data = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291new20180724.xlsx", sheet_name="原始数据")
    beizhu = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291new20180724.xlsx", sheet_name="特征描述")
    cols = list(data.columns)
    "missing>70%"
    cols_drop = ["FT3", "TT3", "TT4"] + ["id", "住院号", "分娩时间"]
    for col in cols_drop:
        cols.remove(col)
    data = data[cols]
    str_col = list(beizhu[beizhu["变量类型"] == "字符串"]["变量"].values)
    num_col = list(beizhu[beizhu["变量类型"] == "数值"]["变量"].values)
    cols_drop = ["FT3", "TT3", "TT4"]
    for col in cols_drop:
        num_col.remove(col)
    data = data[str_col + num_col]

    data.fillna(-1, inplace=True)

    str_col.remove("新生儿是否感染")

    "bool变量要不要 onehot"
    "选出字符串变量中的取值大于2个，bool变量存在缺失值"
    str_col_onehot = []
    for i in str_col:
        if len(set(data[i])) > 2:
            str_col_onehot.append(i)
    for col in str_col_onehot:
        print(col)
        print("*")
        data[col] = data[col].astype(int).astype(str)
    " columns:str_col"
    data = pd.get_dummies(data, columns=str_col, prefix=str_col, prefix_sep="+")
    onehot_cols = data.columns
    need_col = []
    for col in onehot_cols:
        if "+0" not in col:
            need_col.append(col)
    "不标准化，可以直接给数模型"

    data = data_tree = data[need_col]
    print(data_tree.shape)

    "标准化，给线性模型，将连续变量标准化"
    cols_all = list(data.columns)
    cols_str = [i for i in cols_all if i not in num_col]
    data_num = data[num_col]
    data_str = data[cols_str]
    num_col_rbs = data_num.columns
    std = StandardScaler()
    rbs = RobustScaler()
    data_num = rbs.fit_transform(data_num)
    # data_num = std.fit_transform(data_num)
    data_num = pd.DataFrame(data_num, columns=num_col_rbs)
    data_linear = pd.concat([data_num, data_str], axis=1)
    # data_linear.to_csv(r"C:\Users\gang.yu\Desktop\data_linear1.csv",encoding="utf_8_sig")
    return data_linear, data_tree


def sgd_gird(X, y):
    """
{'alpha': 0.0005, 'l1_ratio': 0.99, 'loss': 'log', 'penalty': 'elasticnet'}
auc:0.8465318591928485
{'alpha': 0.0005, 'l1_ratio': 0.9, 'loss': 'log', 'penalty': 'elasticnet'}
auc:0.8291456105925491

{'alpha': 0.0005, 'l1_ratio': 0.9, 'loss': 'log', 'penalty': 'elasticnet'}
auc:0.8238334498262601
{'class_weight': {0: 1, 1: 5}}
auc:0.8434859905403255
    """

    params = {  "alpha": [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
        "l1_ratio": [.01, .1, .5, .9, .99],
        "penalty": ["elasticnet","l2"], #, "l1",
        "class_weight": [{0: 1, 1: 4}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 10},{0: 1, 1: 7},{0: 1, 1: 8}],
        "loss": ["hinge", "log"],
    }
    sgd = SGDClassifier()
    #sgd = SGDClassifier(alpha=0.0005, l1_ratio=0.9, loss="log", penalty='elasticnet', class_weight={0: 1, 1: 5})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    grid = GridSearchCV(sgd, param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=4)
    grid.fit(X, y)
    print(grid.best_params_)
    print("auc:" + str(grid.best_score_))





def logist_grid(X, y):

    """
    {'C': 0.3, 'class_weight': {0: 1, 1: 10}, 'penalty': 'l2'}
    0.8562506709908032
    {'C': 0.3, 'class_weight': {0: 1, 1: 10}, 'penalty': 'l2'}
0.8562506709908032


{'C': 0.8, 'penalty': 'l1'}
0.8613741627503361
    """
    params = {"C": [0.1,0.2,0.3, 0.5, 0.7, 0.8, 0.1, ],
              "penalty": ["l1", "l2", ],
              },
    clf = LogisticRegression(random_state=2018)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    gsearch1 = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc',
                            cv=cv, verbose=1)
    gsearch1.fit(X, y)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    print(gsearch1.cv_results_)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
    metric = np.mean((cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))
    print("cross_val_score :" + str(metric))





def grid_rf(X, y):
    """
{'max_features': 4}
auc_roc0.8731646446513287
{'min_samples_split': 10}
auc_roc0.8721380665529755
{'min_samples_leaf': 13}
auc_roc0.8753863765728656
{'min_samples_split': 5}
auc_roc0.8753863765728656



{'max_depth': 5}
f10.28718858361659466
{'max_features': 3}
f10.2954003207743699
{'min_samples_leaf': 8}
f1:0.30558914648131197
    """
    params = {   'n_estimators':range(40,70,10),
        'max_depth': range(4, 6, 1),
        # 'criterion':["entropy"],
        'max_features': range(6,8),
        # "class_weight":[None,{0: 1, 1:2},{0: 1, 1:6},{0: 1, 1:8},{0: 1, 1:10},{0: 1, 1:50},{0: 1, 1:20},{0: 1, 1:30},{0: 1, 1:40}],
        'min_samples_split': range(6, 9, 1),
         'min_samples_leaf': range(2, 8, 1),

    }
    # {'max_depth': 4, 'n_estimators': 200}
    # roc_auc: 0.8692849885455495
    # {'max_depth': 4, 'n_estimators': 100}
    # roc_auc:0.8695344315522302
    # {'max_depth': 4, 'n_estimators': 70}
    # roc_auc:0.8708626938081094
    # {'max_depth': 4, 'n_estimators': 70}
    # roc_auc:0.8708626938081094
    # {'criterion': 'entropy', 'max_depth': 5, 'n_estimators': 70}
    # roc_auc:0.8755136848893355
    # {'criterion': 'entropy', 'max_depth': 4, 'max_features': 8, 'n_estimators': 50}
    # roc_auc:0.8794755298250266
    # {'max_depth': 4, 'max_features': 8, 'min_samples_split': 7, 'n_estimators': 50}
    # roc_auc:0.8801984634665414

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        criterion="entropy",
        max_features=8,
        # class_weight={0: 1, 1: 10},
        class_weight=None,
        min_samples_split=7,
        min_samples_leaf=2,
        random_state=25,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    grid = GridSearchCV(rf, param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(grid.best_params_)
    print("roc_auc:" + str(grid.best_score_))






def grid_gbdt(X,y):
    """
    {'learning_rate': 0.1, 'n_estimators': 50}
roc_auc:0.8727097424980576
{'max_depth': 5, 'min_samples_split': 300}
roc_auc:0.880121115992192
{'min_samples_leaf': 50, 'min_samples_split': 80}
roc_auc:0.8828240426352606
{'max_features': 5}
roc_auc:0.8839987498317413
{'subsample': 0.9}
roc_auc:0.8873883464142315
{'learning_rate': 0.01, 'n_estimators': 800}
roc_auc:0.8864807262687343
    :param X:
    :param y:
    :return:
    """
    params = {   'n_estimators':[50,100,300,500,800],
                 "learning_rate":[0.01,0.05,0.1,0.3,0.5]
         # 'max_depth': range(3,7, 1),
        #  'min_samples_split': range(50, 300, 30),
        # 'min_samples_leaf': range(30, 101, 10),
        # "subsample":[0.6,0.7,0.8,0.9,1],
       # "max_features":range(3,15,2)
        # 'min_samples_split': range(5, 20, 1),
        #  'min_samples_leaf': range(2, 20, 1),
    }

    gbdt=GradientBoostingClassifier(learning_rate=0.01,
                                    n_estimators=800,
                                    subsample=0.9,
                                    min_samples_split=80,
                                    min_samples_leaf=50,
                                    max_features=5,
                                    max_depth=5,
                                    random_state=2018
                                    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    grid = GridSearchCV(gbdt,param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=4)
    grid.fit(X, y)
    print(grid.best_params_)
    print("roc_auc:" + str(grid.best_score_))


def grid_XGB(X, y):


    params = {  # 'n_estimators':[50,100,300,500,800,1000],
        'max_depth': range(2, 6, 1),
        'min_child_weight': range(1, 6, 2)
    }
    XGB = xgb.XGBClassifier(learning_rate=0.1,
                            n_estimators=500,
                            # max_depth=4,
                            # min_child_weight=0.7,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective='binary:logistic',
                            nthread=4,
                            scale_pos_weight=1,
                            seed=27)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    grid = GridSearchCV(XGB, param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=4)
    grid.fit(X, y)
    print(grid.best_params_)
    print("auc_roc" + str(grid.best_score_))


def cross_val_score1(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1, 10000))
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(cv_scores)
    mean_score1 = np.mean(cv_scores)
    print("test_mean_score:" + str(mean_score1))





def cross_val_score_kfold(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(cv_scores)
    mean_score1 = np.mean(cv_scores)
    print("test_mean_score:" + str(mean_score1))



def sgd_predict(X, y, sgd):
    sgd.fit(X, y)
    feat_importances = sgd.coef_
    feat_importances = list(np.hstack(feat_importances.flat))
    print(len(feat_importances))
    feat_importances = pd.DataFrame({"var": X.columns, "important": feat_importances})
    feat_importances.to_csv("C:/Users/gang.yu/Desktop/gfy_sgd_feat_importances1.csv", encoding="utf_8_sig")




def k_fold_roc_pr(model,X,y):
    X = np.array(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    pres = []
    aucs = []
    mean_recall = np.linspace(0, 1,100)
    i = 0
    all_predict_proba = pd.DataFrame({"probas": []})
    all_predict_label = pd.DataFrame({"label": []})
    for train, test in cv.split(X, y):
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        temp_label = pd.DataFrame({"label": y[test]})
        temp = pd.DataFrame({"probas": probas_[:, 1]})
        all_predict_proba = pd.concat([all_predict_proba, temp])
        all_predict_label = pd.concat([all_predict_label, temp_label])
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
        pres.append(interp(mean_recall[::-1], recall[::-1], precision[::-1]))
        pres[-1][0] = 0.0
        pr_auc = average_precision_score(y[test], probas_[:, 1])
        # roc_auc = auc(recall,precision)

        aucs.append(pr_auc)
        plt.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.2f)' % (i, pr_auc))
        i += 1


   # all_predict_proba.to_csv("C:/Users/gang.yu/Desktop/proba10291_rf.csv", index=False)
    # proba=pd.concat([all_predict_label,all_predict_proba],axis=1)
    # proba.to_csv("C:/Users/gang.yu/Desktop/proba10291_lr.csv", index=False)


    precision, recall, thresholds = precision_recall_curve(all_predict_label["label"], all_predict_proba["probas"])

    fpr, tpr, thresholds_roc = roc_curve(all_predict_label["label"], all_predict_proba["probas"])
    top15_all_thresholds = pd.DataFrame({"thresholds": thresholds})

    top15_all_pr = pd.DataFrame({"precision": precision, "recall": recall})
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

    top15_all_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds_roc})
    "2个坐标文件"
    #top15_all_roc.to_csv("C:/Users/gang.yu/Desktop/top8_roc1.csv", index=False)
    #top15_all_thresholds.to_csv("C:/Users/gang.yu/Desktop/top8_threshold.csv", index=False)
    # top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/top8_pr1.csv", index=False)

    # all_predict_label.to_csv("C:/Users/gang.yu/Desktop/model1_top15_label.csv", index=False)
    # all_predict_proba.to_csv("C:/Users/gang.yu/Desktop/model1_top15_predict_proba.csv", index=False)
    # print(all_predict_proba)
    print("mean_auc:" + str(np.mean(aucs)))
    mean_pre = np.mean(pres, axis=0)
    mean_pre[-1] = 1.0

    "只有平均的recall,precision,AP需要标签和概率"
    #mean_auc = average_precision_score()
    mean_auc = auc(mean_recall, mean_pre)
    std_auc = np.std(aucs)
    # model1_top15_roc_df=pd.DataFrame({"mean_recall":mean_recall,"mean_pre":mean_pre[::-1]})
    # model1_top15_roc_df.to_csv("C:/Users/gang.yu/Desktop/gfy_top15_pr.csv",index=False)
    plt.plot(mean_recall, mean_pre[::-1], color='b',
             label=r'Mean PR (AUC = %0.3f $\pm$ %0.2f))' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_pre = np.std(pres, axis=0)
    pres_upper = np.minimum(mean_pre + std_pre, 1)
    pres_lower = np.maximum(mean_pre - std_pre, 0)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('')
    plt.xlabel('Recall,%')
    plt.ylabel('Precision,%')
    plt.legend(loc="upper right")
    plt.figure()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
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
    interval = stats.t.interval(0.95, 4, np.mean(aucs), std_auc)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('1-Specificity,%')
    plt.ylabel('Sencitivity,%')
    plt.legend(loc="lower right")
    plt.show()

    # plt.fill_between(mean_recall, pres_lower[::-1], pres_upper[::-1], color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')  #  interval=(%0.2f,%0.2f)   % (interval[0],interval[1])

    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')#  interval=(%0.2f,%0.2f) % (interval[0],interval[1]))



def gbdt_lr(X, y):
    # 弱分类器的数目
    n_estimator = 50
    # 随机生成分类数据。
    # 切分为测试集和训练集，比例0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=0.3)
    # 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, shuffle=True, stratify=y_train,
                                                               test_size=0.5)
    # 调用GBDT分类模型。
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    # 调用one-hot编码。
    grd_enc = OneHotEncoder()
    # 调用LR分类模型。
    grd_lm = LogisticRegression()
    '''使用X_train训练GBDT模型，后面用此模型构造特征'''
    grid_gbdt(X_train,y_train)




    #grd.fit(X_train,y_train)








    #
    # # fit one-hot编码器
    # grd_enc.fit(grd.apply(X_train)[:, :, 0])
    #
    #
    #
    # '''
    # 使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
    # '''
    # grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    # # 用训练好的LR模型多X_test做预测
    # y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    # # 根据预测结果输出
    #
    # fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_grd_lm)
    # # fpr_train, tpr_train, thresholds_train = roc_curve(y_train,pre_train)
    # # roc_auc_train = auc(fpr_train, tpr_train)
    # # plt.plot(fpr_train, tpr_train, lw=2, alpha=0.8,color="b",
    # #          label='(train AUC = %0.3f)' % ( roc_auc_train))
    # roc_auc_test = auc(fpr_test, tpr_test)
    # plt.plot(fpr_test, tpr_test, lw=2, alpha=0.8, color="r",
    #          label='(test AUC = %0.3f)' % (roc_auc_test))
    # plt.legend(loc="lower right")
    # plt.xlabel('1-Specificity,%')
    # plt.ylabel('Sencitivity,%')
    # plt.figure()
    # "pr-AP"
    # precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, y_pred_grd_lm)
    # # precision_train, recall_train, thresholds_train = precision_recall_curve(y_train,pre_train)
    # AP_test = average_precision_score(y_test, y_pred_grd_lm)
    # # AP_train = average_precision_score(y_train,pre_train)
    # plt.plot(recall_test, precision_test, lw=2, alpha=0.8, color="b",
    #          label='(test AP = %0.3f)' % (AP_test))
    #
    # # plt.plot(recall_train, precision_train, lw=2, alpha=0.8,color="r",
    # #          label='(train AP = %0.3f)' % (AP_train))
    # plt.legend(loc="lower left")
    # plt.xlabel('Recall,%')
    # plt.ylabel('Precision,%')
    # plt.show()




def model_predict73_compare_train_test_important(model, X_tree, y_tree):
    X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, shuffle=True, test_size=0.33,
                                                        random_state=2018)
    model.fit(X_train, y_train)
    # model_feat_importace = pd.DataFrame({"feat": X_train.columns.values, "scores": model.feature_importances_})
    # model_feat_importace.sort_values(by="scores", ascending=False, inplace=True)
    # #model_feat_importace.to_csv("C:/Users/gang.yu/Desktop/gfy_xgb_feat_importances.csv", encoding="utf_8_sig")
    pre_test = model.predict_proba(X_test)[:,1]
    pre_train = model.predict_proba(X_train)[:,1]

    data=pd.DataFrame({"y_test":y_test,"pre_test":pre_test})
    data.to_csv("C:/Users/gang.yu/Desktop/lr.csv", encoding="utf_8_sig")

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pre_test)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, pre_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, lw=2, alpha=0.8, color="b",
             label='(train AUC = %0.3f)' % (roc_auc_train))
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, lw=2, alpha=0.8, color="r",
             label='(test AUC = %0.3f)' % (roc_auc_test))
    plt.legend(loc="lower right")
    plt.xlabel('1-Specificity,%')
    plt.ylabel('Sencitivity,%')
    plt.figure()

    print(classification_report(y_test, model.predict(X_test)))
    "pr-AP"
    precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, pre_test)
    precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, pre_train)
    AP_test = average_precision_score(y_test, pre_test)
    AP_train = average_precision_score(y_train, pre_train)
    plt.plot(recall_test, precision_test, lw=2, alpha=0.8, color="b",
             label='(test AP = %0.3f)' % (AP_test))

    plt.plot(recall_train, precision_train, lw=2, alpha=0.8, color="r",
             label='(train AP = %0.3f)' % (AP_train))

    plt.legend(loc="lower left")
    plt.xlabel('Recall,%')
    plt.ylabel('Precision,%')
    plt.show()
def undersampling(data_linear):
    "先73分，7用于训练"
    print("undersampling")
    # y_tree = data_tree["新生儿是否感染"]
    # X_tree = data_tree.drop("新生儿是否感染", axis=1)
    y_tree = data_linear["新生儿是否感染"]
    X_tree = data_linear.drop("新生儿是否感染", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, shuffle=True, test_size=0.3,
                                           random_state=10)
    data_train=pd.concat([X_train,y_train],axis=1)
    data_train_neg = data_train[data_train["新生儿是否感染"] == 0]
    data_train_pos = data_train[data_train["新生儿是否感染"] == 1]
    data_train_neg=shuffle(data_train_neg)
    "10:1"
    data_train_neg_0=data_train_neg[:2000]

    data_train_sample=pd.concat([data_train_neg_0,data_train_pos],axis=0)

    y_sample = data_train_sample["新生儿是否感染"]
    X_sample = data_train_sample.drop("新生儿是否感染", axis=1)
    logist_grid(X_sample, y_sample)
    #cross_val_score_kfold(lr,X_sample, y_sample)



    # model=xgb.XGBClassifier(
    #     max_depth=3, learning_rate=0.01,
    #     n_estimators=500, silent=True,
    #     objective="binary:logistic", booster='gbtree',
    #     n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
    #     max_delta_step=0, subsample=0.7, colsample_bytree=0.8, colsample_bylevel=0.8,
    #     reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
    # )

    model.fit(X_train, y_train)
    pre_test = model.predict_proba(X_test)[:, 1]
    pre_train = model.predict_proba(X_train)[:, 1]
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pre_test)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, pre_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, lw=2, alpha=0.8, color="b",
             label='(train AUC = %0.3f)' % (roc_auc_train))
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, lw=2, alpha=0.8, color="r",
             label='(test AUC = %0.3f)' % (roc_auc_test))
    plt.legend(loc="lower right")
    plt.xlabel('1-Specificity,%')
    plt.ylabel('Sencitivity,%')
    plt.figure()
    "pr-AP"
    precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, pre_test)
    precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, pre_train)
    AP_test = average_precision_score(y_test, pre_test)
    AP_train = average_precision_score(y_train, pre_train)
    plt.plot(recall_test, precision_test, lw=2, alpha=0.8, color="b",
             label='(test AP = %0.3f)' % (AP_test))
    plt.plot(recall_train, precision_train, lw=2, alpha=0.8, color="r",
             label='(train AP = %0.3f)' % (AP_train))
    plt.legend(loc="lower left")
    plt.xlabel('Recall,%')
    plt.ylabel('Precision,%')
    plt.show()




def svc_grid(X,y):
    """
    {'C': 0.5, 'class_weight': {0: 1, 1: 10}, 'kernel': 'rbf'}
Traceback (most recent call last):
roc_auc:0.8730569733545431
    :param X:
    :param y:
    :return:
    """
    print("********svc grid********")
    params = {'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'class_weight': [None, {0: 1, 1:6}, {0: 1, 1: 2},
                               {0: 1, 1: 10}, {0: 1, 1: 15}, {0: 1, 1: 20}]

    }
    svc=SVC()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    grid = GridSearchCV(svc, param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(grid.best_params_)
    print("roc_auc:" + str(grid.best_score_))








class Ensemble(object):

    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2018).split(X, y))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_splits))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                "预测概率"
                y_pred = clf.predict_proba(X_holdout)[:, 1]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)
        "训练第二层lr"
        self.stacker.fit(S_train,y)
        res = self.stacker.predict_proba(S_test)[:,1]
        res_label= self.stacker.predict(S_test)[:]
        return res,res_label

def stacking(data_stacking):
    y =data_stacking["新生儿是否感染"]
    X =data_stacking.drop("新生儿是否感染", axis=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.33,
                                                        random_state=2018)
    svc=SVC(C= 0.5, class_weight={0: 1, 1: 10}, kernel='rbf',probability=True,random_state=2018)
    rf=RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        max_features=3,
        class_weight={0: 1, 1: 10},
        min_samples_split=5,
        min_samples_leaf=8,
        random_state=10,

    )
    gbdt = GradientBoostingClassifier(learning_rate=0.01,
                                      n_estimators=800,
                                      subsample=0.9,
                                      min_samples_split=80,
                                      min_samples_leaf=50,
                                      max_features=5,
                                      max_depth=5,
                                      random_state=2018
                                      )
    stack = Ensemble(n_splits=5,
                     stacker=LogisticRegression(),
                     base_models=(gbdt,rf,svc))
    pre_test,res_label=stack.fit_predict(X_train,y_train,X_test)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pre_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, lw=2, alpha=0.8, color="r",
             label='(test AUC = %0.3f)' % (roc_auc_test))
    plt.legend(loc="lower right")
    plt.xlabel('1-Specificity,%')
    plt.ylabel('Sencitivity,%')
    plt.figure()

    #print(classification_report(y_test, res_label))
    "pr-AP"
    precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, pre_test)
    AP_test = average_precision_score(y_test, pre_test)

    plt.plot(recall_test, precision_test, lw=2, alpha=0.8, color="b",
             label='(test AP = %0.3f)' % (AP_test))
    plt.legend(loc="lower left")
    plt.xlabel('Recall,%')
    plt.ylabel('Precision,%')
    plt.show()


    return y_test,result








if __name__ == '__main__':
    print("*********start*********")
    data_linear, data_tree = preprocessing_for_sgd()
    for i in data_linear.columns:
        print(i)
    y = data_linear["新生儿是否感染"]
    X = data_linear.drop("新生儿是否感染", axis=True)


    # data = pd.read_excel(r"C:\Users\gang.yu\Desktop\10291new20180724.xlsx", sheet_name="sgd_feat")
    # col=list(data[data["val"]>0]["var"])
    # X_top=X[col]
    y_tree = data_tree["新生儿是否感染"]
    X_tree = data_tree.drop("新生儿是否感染", axis=True)
    cols = ["关联CRP+-1", "关联CRP+1", "中性粒细胞比例", "白细胞计数", "产程开始_胎膜早破时间差_h",
            "胎膜破裂_分娩时间差_h", "孕周_足月为0_近足月早产为1_远足月早产为2+1",
            "B族链球菌+-1", "TSH", "FT4", "血小板计数", "体温", "ogtt", "height", "B族链球菌+1",
            "产妇年龄", "羊水情况_清为0_混为1_胎粪为2+2",
            ]

    X_tree_top = X_tree[cols]

    cols_stacking=["关联CRP+1","关联CRP+-1","孕周_足月为0_近足月早产为1_远足月早产为2+1","B族链球菌+-1","B族链球菌+1",
                   "羊水情况_清为0_混为1_胎粪为2+2","产后5分钟Apgar评_8到10分为0_4到7分为1_0到3分为2+1","产后5分钟Apgar评_8到10分为0_4到7分为1_0到3分为2+-1",
                   "产后5分钟Apgar评_8到10分为0_4到7分为1_0到3分为2+2",
                   "白细胞计数","产后1分钟Apgar评_8到10分为0_4到7分为1_0到3分为2+1","产后1分钟Apgar评_8到10分为0_4到7分为1_0到3分为2+-1",
                   "产后1分钟Apgar评_8到10分为0_4到7分为1_0到3分为2+2",
                   "产程开始_胎膜早破时间差_h",
                   "羊水情况_清为0_混为1_胎粪为2+-1","羊水情况_清为0_混为1_胎粪为2+1",
                   "是否抗生素+1","中性粒细胞比例","TSH","胎膜破裂_分娩时间差_h","生产次数+2",
                   "生产次数+3","生产次数+4",
                   "分娩方式_自产为0_产钳为1_剖腹产为2+2",
                   "孕周_足月为0_近足月早产为1_远足月早产为2+1"
                ]
    cols_stacking.append("新生儿是否感染")
    data_stacking=data_linear[cols_stacking]

    #sgd_gird(X, y)
    # "特征排序 "
    # sgd = SGDClassifier(alpha=0.0005, l1_ratio=0.9, loss="log", penalty='elasticnet', class_weight={0: 1, 1: 5})
    "bool值保留"
    # {'alpha': 0.001, 'class_weight': {0: 1, 1: 8}, 'l1_ratio': 0.99, 'loss': 'hinge', 'penalty': 'elasticnet'}
    # auc: 0.8557192367684917
    # sgd = SGDClassifier(alpha=0.001, l1_ratio=0.99, loss="hinge", penalty='elasticnet', class_weight={0: 1, 1: 8})
    # sgd_predict(X,y,sgd)
    #grid_rf(X_tree_top,y_tree)
    #用全部特征
    #grid_XGB(X_tree,y_tree)
    "系数大于0的特征 "
   # lr=LogisticRegression(C=0.3,class_weight={0: 1, 1:10},penalty="l2",random_state=2014)

    lr = LogisticRegression(C=0.8, penalty="l1", random_state=2014)

    rf = RandomForestClassifier(
        # n_estimators=50,
        # max_depth=4,
        # criterion="entropy",
        # max_features=8,
        # # class_weight={0: 1, 1: 10},
        # class_weight=None,
        # min_samples_split=7,
        # min_samples_leaf=2,
        # random_state=25,

            #"从下到上"
            # n_estimators=300,
            #  max_depth=3,
            # max_features=4,
            # class_weight={0: 1, 1:40},
            # min_samples_split=15,
            # min_samples_leaf=13,
            # random_state=10,

       #原始
        # n_estimators=800,
        # max_depth=4,
        # class_weight={0: 1, 1: 10},
        # min_samples_split=5,
        # min_samples_leaf=2,
        # random_state=10,

        #从下到auc，从上到下f1
        n_estimators=500,
        max_depth=5,
        max_features=3,
        class_weight={0: 1, 1: 10},
        min_samples_split=5,
        min_samples_leaf=8,
        random_state=10,
    )
    #stacking(data_stacking)







    model_predict73_compare_train_test_important(lr,X_tree_top, y_tree)


    #k_fold_roc_pr(lr,X_top, y)
    #k_fold_roc_pr(rf, X_tree_top, y_tree)
    #undersampling(data_tree)
    #cross_val_score1(rf, X_tree_top, y_tree)
    #cross_val_score1(lr, X, y)
    #logist_grid(X, y)



    #gbdt_lr(X_tree_top, y_tree)

    #model_predict73_compare_train_test_important(rf, X_tree, y_tree)
    # sgd_predict(X, y,sgd)
    "调参"
    XGB = xgb.XGBClassifier(learning_rate=0.1,
                                n_estimators=500,
                                max_depth=4,
                                min_child_weight=0.7,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                objective='binary:logistic',
                                nthread=4,
                                scale_pos_weight=1,
                                seed=27)
