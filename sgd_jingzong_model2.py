from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold, StratifiedKFold
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNet, ElasticNetCV, Lasso, SGDClassifier
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score,precision_recall_curve
import lightgbm as lgt
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from scipy import interp
from sklearn_ml_core.Lib.CurveLib import CurveLib
from hyperopt import tpe
from sklearn.metrics import auc, roc_curve, classification_report
from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel("D:/info_data/jingzong/补全缺失值数据分析_new.xlsx")
describe_analysis = pd.read_excel("D:/info_data/jingzong/补全缺失值数据分析_new.xlsx", sheet_name="描述分析")
print(data.shape)
# 90个p<=0.05
val = list(describe_analysis[((describe_analysis["P "] < 0.05) | (describe_analysis["P "] == 0.05))]["变量名称"])
val.append("PSP二分类(1)")

data_drop_p = data[val]

data_drop_p["吸烟总计"].fillna(data_drop_p["吸烟总计"].mean(), inplace=True)

drop_null_cols = ["淋巴细胞比例", "单核细胞比例", "AST", "尿素氮", "糖化血红蛋白", "LDL", "血清催乳素_1","Ⅰ、疾病严重度CGIS_1"]  #
data_drop_p.drop(drop_null_cols, axis=1, inplace=True)
d=data_drop_p.isnull().sum()

#d.to_csv("C:/Users/gang.yu/Desktop/null.csv",encoding="utf_8_sig")

def str_to_onehot(data_drop_p):
    one_hot_cols=["职业","类别","发病情况","家庭支持","社区支持"]     #bool:"是否有吸烟习惯","陪诊情况","氯氮平","氨磺必利","阿立哌唑"
    for col in one_hot_cols:
        data_drop_p[col]=data_drop_p[col].astype(str)

    data_drop_p=pd.get_dummies(data_drop_p)
    return data_drop_p

#data_drop_p=str_to_onehot(data_drop_p)

def get_topk_features():
    sgd_importances = pd.read_excel("D:/info_data/jingzong/补全缺失值数据分析_new.xlsx", sheet_name="model2_importances_feat")
    #topk_importance = list(sgd_importances[sgd_importances["val"] > 0.0627]["var"])
    topk_importance = list(sgd_importances[:8]["var"])
    print(len(topk_importance ))
    #topk_importance.remove("血清催乳素_1")
    topk_importance.append("PSP二分类(1)")
    print(topk_importance)
    return data_drop_p[topk_importance]


data_topk = get_topk_features()

#data_topk.to_csv("C:/Users/gang.yu/Desktop/pre.csv",encoding="utf_8_sig")


y = data_topk["PSP二分类(1)"]
X = data_topk.drop("PSP二分类(1)", axis=1)

print(X.shape)
print(y.shape)


# y=data_drop_p["PSP二分类(1)"]
# X=data_drop_p.drop("PSP二分类(1)",axis=1)






"具体指定些变化大的变量归一化 "
std = StandardScaler()
column = X.columns


"随机森林不需要std"
print(len(column))
#X = std.fit_transform(X)


from functools import partial
from hyperopt import hp, fmin, tpe


def grid_model_lgt(X, y):
    params = {"n_estimators": [50, 100, 300, 500, 800],
              "learning_rate": [.01, .05, .1, .5, ],
              # 'max_depth':[3,5,8],
              # "max_bin":[30,50,20],
              # 'feature_fraction':[0.7,0.8,0.9,1],
              # 'bagging_fraction':[0.7,0.8,0.9,1]
              # "n_iter":[100,500,800],
              }
    lgt = LGBMClassifier(num_leaves=50,
                         max_depth=13,
                         learning_rate=0.1,
                         n_estimators=1000,
                         min_child_weight=1,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         nthread=7, )  #
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    grid = GridSearchCV(lgt, param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)


def lgt_raw_using(X, y, test_X, test_y):
    lgb_train = lgt.Dataset(X, y)
    lgb_eval = lgt.Dataset(test_X, test_y, reference=lgb_train)
    params = {"objective": 'binary',
              'max_depth': 5,
              'num_leaves': 64,
              # "num_leaves":5,   #num_leaves = 2^(max_depth)
              "learning_rate": 0.05,
              "n_estimators": 300,
              "max_bin": 55,
              "bagging_fraction": 0.8,
              "bagging_freq": 5,
              "feature_fraction": 0.2319,
              "feature_fraction_seed": 9,
              "bagging_seed": 9,
              "min_data_in_leaf": 6,
              "min_sum_hessian_in_leaf": 11,
              }
    model = lgt.train(params=params, train_set=lgb_train, early_stopping_rounds=10,
                      valid_sets=lgb_eval, )

    result = model.predict(X, num_iteration=model.best_iteration, pred_leaf=True)  # num_iteration=clf.best_iteration
    result_test = model.predict(test_X, num_iteration=model.best_iteration, )
    result_test = [1 if i > 0.4 else 0 for i in result_test]
    # print(result_test)
    # print(test_y.shape)
    # feat_important=model.feature_importance
    # feat_importances=pd.DataFrame({"var":X.columns,"important":feat_important})
    # feat_importances.to_csv("C:/Users/gang.yu/Desktop/lgt_feat_importances.csv",encoding="utf_8_sig")
    # compare=pd.DataFrame({"real":test_y,"pre":result_test})
    # compare.to_csv("C:/Users/gang.yu/Desktop/compare.csv")
    print("lgt：%f" % roc_auc_score(test_y, result_test))
    return model, result, result_test


# lgt_save,lgt_result,result_test=lgt_raw_using(train_X,train_y,test_X,test_y)





def many_clf_hypert():
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=np.random.randint(1, 100))
    estim = HyperoptEstimator(classifier=any_classifier('clf'), algo=tpe.suggest)
    estim.fit(train_X, train_y)
    print(estim.score(test_X, test_y))
    print(estim.best_model())


def sgd_hyperopt():
    space = {"alphas": hp.choice("alphas", [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]),
             "l1_ratio": hp.choice("l1_ratio", [.01, .1, .5, .9, .99]),
             "loss": hp.choice("loss", ["hinge", "log"]),
             "n_iter": hp.randint("n_iter", 500),
             # "class_weight":hp.choice("class_weight",)
             }

    def sgd(args):
        sgd = SGDClassifier(alpha=args["alphas"], average=False, class_weight=None, epsilon=0.1,
                            eta0=0.0, fit_intercept=True, l1_ratio=args["l1_ratio"],
                            learning_rate='optimal', loss='hinge', n_iter=args["n_iter"], n_jobs=1,
                            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
                            verbose=0, warm_start=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        metric = np.mean((cross_val_score(sgd, X, y, cv=cv, scoring="roc_auc")))
        print(metric)
        return -metric

    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(sgd, space, algo=algo, max_evals=20)
    print(best)
    print(sgd(best))


# sgd_hyperopt()



def grid_model(X, y):
    params = {"alpha": [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
              "l1_ratio": [.01, .1, .5, .9, .99],
              "penalty": ["l1", "l2", "elasticnet"],
              "class_weight": [{0: 1, 1: 4}, {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 7}]
              # "loss":["hinge","log"],
              # "n_iter":[100,500,800],
              }
    sgd = SGDClassifier(loss="hinge")  #
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1,100))
    grid = GridSearchCV(sgd, param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_params_)
    print("roc_auc:" + str(grid.best_score_))


"""
#loss=hinge
{'alpha': 1, 'class_weight': {0: 1, 1: 3}, 'l1_ratio': 0.1, 'penalty': 'l2'}
roc_auc:0.8578479070118307

{'alpha': 1, 'class_weight': {0: 1, 1: 3}, 'l1_ratio': 0.1, 'penalty': 'l2'}
roc_auc:0.8635415676044909



#loss=log
{'alpha': 1, 'class_weight': {0: 1, 1: 7}, 'l1_ratio': 0.01, 'penalty': 'l2'}
roc_auc:0.8597886942928576
{'alpha': 1, 'class_weight': {0: 1, 1: 7}, 'l1_ratio': 0.01, 'penalty': 'l2'}
roc_auc:0.8567231724644614
{'alpha': 1, 'class_weight': {0: 1, 1: 7}, 'l1_ratio': 0.01, 'penalty': 'elasticnet'}
roc_auc:0.8575072348934148

"""

# sgd=SGDClassifier(loss="hinge", penalty='l2', alpha=0.1,l1_ratio=0.01, class_weight={0: 1, 1:4})

"提取特征"
sgd = SGDClassifier(loss="hinge", penalty='l2', alpha=1, l1_ratio=0.1, class_weight={0: 1, 1:3})



"很多0不好选"
# sgd=SGDClassifier(loss="log", penalty='elasticnet', alpha=0.1,l1_ratio=0.5, class_weight={0: 1, 1:6})


# 偏好
def sgd_predict():
    sgd.fit(X, y)
    feat_importances = sgd.coef_
    feat_importances = list(np.hstack(feat_importances.flat))
    print(len(feat_importances))
    feat_importances = pd.DataFrame({"var": column, "important": feat_importances})
    feat_importances.to_csv("C:/Users/gang.yu/Desktop/model2_sgd_feat_importances5.csv", encoding="utf_8_sig")


def cross_val_score1(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1, 100), )
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc",verbose=4)
    print(cv_scores)
    mean_score1 = np.mean(cv_scores)
    print("test_mean_score:" + str(mean_score1))


def train_test_sgd():
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=np.random.randint(1, 100))
    sgd.fit(train_X, train_y)
    print(train_y.sum() / train_y.count())
    pre_test = sgd.predict(test_X)
    # pre_test_score = sgd.predict_proba(test_X)
    # pre_train_score = sgd.predict_proba(train_X)
    # draw_pr(train_y,pre_train_score,  test_y,pre_test_score)
    print("test scores:" + str(roc_auc_score(test_y, pre_test)))
    return


def draw_ROC_curve(y_test, y_predict, train_y, pre_train_score):
    '''''
    画ROC曲线
    '''
    plt.figure()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    false_positive_rate_train, true_positive_rate_train, thresholds_train = roc_curve(train_y, pre_train_score)

    roc_auc_train = auc(false_positive_rate_train, true_positive_rate_train)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.plot(false_positive_rate_train, true_positive_rate_train, 'r', label='AUC = %0.2f' % roc_auc_train)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.savefig("C:/Users/gang.yu/Desktop/rf")
    plt.show()


def draw_pr(y_train_true, y_train_score, y_test_true, y_test_score):
    # Plot PR curve
    if y_test_true is not None and y_test_score is not None:
        precision_test, recall_test, auc_test = CurveLib.get_pr(y_test_score, np.array(y_test_true))

    if y_train_true is not None and y_train_score is not None:
        precision_train, recall_train, auc_train = CurveLib.get_pr(y_train_score, np.array(y_train_true))
    plt.figure(figsize=(10, 6))

    plt.plot(recall_test, precision_test, label="test (AUC: {:.3f})".format(auc_test), linewidth=2)
    plt.plot(recall_train, precision_train, label="train (AUC: {:.3f})".format(auc_train), linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()


def grid_rf(X, y):
    params = {  # 'n_estimators':[50,100,300,500,800],
        #'max_depth': range(3, 14, 2)
       # 'min_samples_split': range(2,40,  2),
        'min_samples_leaf': range(2, 30,2),
        # 'max_features': range(3, 8, 2),
        # "class_weight":[{0: 1, 1:4},{0: 1, 1:3},{0: 1, 1:5},{0: 1, 1:6},{0: 1, 1:7}]

        # 'min_samples_split': [3,5, 10],
        # 'min_samples_leaf': [2,4, 8],

    }
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight={0: 1, 1: 4},
         random_state=10,

    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1,100))
    grid = GridSearchCV(rf, param_grid=params, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_params_)
    print("auc_roc" + str(grid.best_score_))

#
# rf = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=5,
#        class_weight={0: 1, 1:4},
#         random_state=10)






def train_test_rf():

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, stratify=y,
                                                        random_state=np.random.randint(1, 100))
    column_trian=train_X.columns
    column_test = test_X.columns
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,  # 9  5
        class_weight={0: 1, 1: 4},
        random_state=10)

    rf.fit(train_X, train_y)
    print(train_y.sum() / train_y.count())
    pre_test = rf.predict(test_X)
    pre_train = rf.predict(train_X)

    print(classification_report(test_y, pre_test))
    feat_importances = rf.feature_importances_
    feat_importances = pd.DataFrame({"var": column, "important": feat_importances})
    #feat_importances.to_csv("C:/Users/gang.yu/Desktop/rf_feat_importances10.csv", encoding="utf_8_sig")
    pre_test_score = rf.predict_proba(test_X)[:, 1]

    pre_train_score = rf.predict_proba(train_X)[:, 1]

    #train_X = std.inverse_transform(train_X)

    train_X_df = pd.DataFrame(train_X.values, columns=column_trian)
    label_index=pd.DataFrame({"id":list(train_y.index),"y_true":train_y.values})
    pre_train_df = pd.DataFrame(pre_train, columns=["pre_label"])
    pre_train_score_df = pd.DataFrame(pre_train_score, columns=["pre_score"])
    train_top = pd.concat([label_index,pre_train_df, pre_train_score_df,train_X_df], axis=1)


    #test_X = std.inverse_transform(test_X)
    test_X_df = pd.DataFrame(test_X.values, columns=column_test)
    label_index = pd.DataFrame({"id": list(test_y.index), "y_true": test_y.values})
    pre_test_df = pd.DataFrame(pre_test, columns=["pre_label"])
    pre_test_score_df = pd.DataFrame(pre_test_score, columns=["pre_score"])
    test_top = pd.concat([label_index, pre_test_df, pre_test_score_df,test_X_df], axis=1)
    print(train_top[:10])


    # test_top.to_csv("C:/Users/gang.yu/Desktop/model2_test_top15.csv", encoding="utf_8_sig",index=False)
    # train_top.to_csv("C:/Users/gang.yu/Desktop/model2_train_top15.csv", encoding="utf_8_sig",index=False)

    print("test auc scores:" + str(roc_auc_score(test_y, pre_test_score)))



    # draw_pr( train_y, pre_train_score,test_y,pre_test_score)
    # draw_ROC_curve(test_y, pre_test_score, train_y, pre_train_score)


def cross_val_score1(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1, 100), )
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(cv_scores)
    mean_score1 = np.mean(cv_scores)
    print("test_mean_score:" + str(mean_score1))


def rf_molel():
    rf = RandomForestClassifier(oob_score=True)
    rf.fit(X, y)
    y_predprob = rf.predict_proba(X)[:, 1]
    print("auc" + str(roc_auc_score(y, y_predprob)))
    print(rf.oob_score_)

def k_fold_roc_pr(X,y):
    X=np.array(X)
    cv = StratifiedKFold(n_splits=5,random_state=2018)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,  # 9  5
        class_weight={0: 1, 1: 4},
        random_state=10)


    pres = []
    aucs = []
    all_predict_proba = pd.DataFrame({"probas": []})
    all_predict_label = pd.DataFrame({"label": []})
    mean_recall = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = rf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        temp_label = pd.DataFrame({"label": y[test]})
        temp = pd.DataFrame({"probas": probas_[:, 1]})
        all_predict_proba = pd.concat([all_predict_proba, temp])
        all_predict_label = pd.concat([all_predict_label, temp_label])

        precision, recall, _ = precision_recall_curve(y[test], probas_[:, 1])

        pres.append(interp(mean_recall[::-1],recall[::-1], precision[::-1]))
        print(pres)
        pres[-1][0] = 0.0
        roc_auc = auc(recall,precision)
        aucs.append(roc_auc)
        plt.plot(recall,precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Luck', alpha=.8)
    print("mean_auc:"+str(np.mean(aucs)))
    mean_pre = np.mean(pres, axis=0)
    mean_pre[-1] = 1.0
    "d "
    precision, recall, thresholds = precision_recall_curve(all_predict_label["label"], all_predict_proba["probas"])
    fpr, tpr, thresholds_roc = roc_curve(all_predict_label["label"], all_predict_proba["probas"])
    print(precision.shape)
    print(recall.shape)
    print(thresholds.shape)
    top15_all_thresholds = pd.DataFrame({"thresholds": thresholds})
    top15_all_pr = pd.DataFrame({"precision": precision, "recall": recall})
    top15_all_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds_roc})
    top15_all_roc.to_csv("C:/Users/gang.yu/Desktop/model2_top15_roc1.csv", index=False)
    # ,"thresholds":thresholds})
    top15_all_thresholds.to_csv("C:/Users/gang.yu/Desktop/model2_top15_threshold.csv", index=False)
    top15_all_pr.to_csv("C:/Users/gang.yu/Desktop/model2_top15_pr1.csv", index=False)

    # all_predict_label.to_csv("C:/Users/gang.yu/Desktop/model1_top15_label.csv", index=False)
    # all_predict_proba.to_csv("C:/Users/gang.yu/Desktop/model1_top15_predict_proba.csv", index=False)

    mean_auc = auc(mean_recall, mean_pre)

    print("mean_auc"+str(mean_auc))

    std_auc = np.std(aucs)
    interval = stats.t.interval(0.95, 4, np.mean(aucs),std_auc)
    print(interval)
    print("mean_recall")
    print(mean_recall)
    print("mean_pre")
    print(mean_pre)
    plt.plot(mean_recall, mean_pre[::-1], color='b',
             label=r'Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_pre= np.std(pres, axis=0)
    pres_upper = np.minimum(mean_pre + std_pre, 1)
    pres_lower = np.maximum(mean_pre - std_pre, 0)
    plt.fill_between(mean_recall, pres_lower[::-1], pres_upper[::-1], color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')  #  interval=(%0.2f,%0.2f)   % (interval[0],interval[1])

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision recall')
    plt.legend(loc="lower left")







    plt.figure()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = rf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        print("fpr")
        print(fpr)
        print("tpr")
        print(tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        print("tprs")
        print(tprs)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    interval = stats.t.interval(0.95,4,np.mean(aucs) ,std_auc)
    print(interval)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')#  interval=(%0.2f,%0.2f) % (interval[0],interval[1]))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()




if __name__ == '__main__':
    # grid_model_lgt(X, y)
    #train_test_sgd()
    k_fold_roc_pr(X,y)
    #cross_val_score1(sgd, X, y)
   # grid_rf(X, y)
    #train_test_rf()
    # sgd_predict()
   # cross_val_score1(rf, X, y)
    #sgd_predict()
    #grid_model(X, y)
    #print("end")