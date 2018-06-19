# coding=utf-8
import numpy as np
import pandas as pd
import numpy
import pydot
import sklearn
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, IsolationForest, \
    BaggingRegressor
from sklearn.linear_model import LogisticRegression, ElasticNetCV, SGDClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn_ml_core.DataAdapter.Excel.ExcelAdapter import ExcelAdapter
from sklearn_ml_core.DataAdapter.ResearchDataMetaData import ResearchDataRawColumnType
from sklearn_ml_core.Lib.CurveLib import CurveLib
from sklearn_ml_core.Lib.CustomLinearRegression import CustomLinearRegression
from sklearn_ml_core.Lib.DelayPrint import DelayPrint
from sklearn_ml_core.Lib.Pvalue import Pvalue
from sklearn_ml_core.Lib.ToolLib import ToolLib
from sklearn_ml_core.Lib.tree_post_prune import tree_post_prune
from sklearn_ml_core.Linq.Linq import Linq
from sklearn_ml_core.Lib.Report import Report

def get_LogisticRegression_expression(rank_data, format_number=str):
    return "y=" + "".join(map(lambda x:
                              ("-" if x.get("rank") < 0 else "+")
                              + format_number(x.get("rank"))
                              + "*" + x.get("x"), rank_data))


def fit(algorithm, _x, _y, descriptions):
    algorithm.fit(_x, _y)
    result = None
    if hasattr(algorithm, "feature_importances_"):
        result = sorted([{'desc': descriptions[index].name, 'rank': item} for index, item in
                         enumerate(algorithm.feature_importances_)], key=lambda x: x["rank"], reverse=True)
    elif hasattr(algorithm, "coef_"):
        result = sorted(
            [{'x': descriptions[index].name, 'rank': item} for index, item in enumerate(algorithm.coef_[0])],
            key=lambda x: x["rank"], reverse=True)
        report.get_cursor().remark.append(get_LogisticRegression_expression(result))
        report.get_cursor().remark.append(get_LogisticRegression_expression(result, lambda x: str(round(x, 2))))

    report.get_cursor().feature_rank = result
    delay_print.delay_print(result)
    if result is None:
        delay_print.delay_print("cannot calculate rank (coef, feature importances or others) now")


def get_recall(y, metrics_confusion_matrix):
    recall0 = metrics_confusion_matrix[0][0] / (metrics_confusion_matrix[0][0] + metrics_confusion_matrix[0][1])
    recall1 = metrics_confusion_matrix[1][1] / (metrics_confusion_matrix[1][0] + metrics_confusion_matrix[1][1])
    return recall0, recall1


def delay_print_predicate_result(y, result):
    delay_print.delay_print(metrics.classification_report(y, result))
    metrics_confusion_matrix = metrics.confusion_matrix(y, result)
    recall0, recall1 = get_recall(y, metrics_confusion_matrix)
    # if recall0<0.5 or recall1<0.5:
    # delay_print.will_print = False
    delay_print.delay_print(metrics.confusion_matrix(y, result))
    return recall0, recall1, metrics_confusion_matrix


def predicate(algorithm, x, y = None):
    result = list(map(g_cut_off_lambda,algorithm.predict_proba(x).tolist()))

    return result




def predicate_train_data(clf, x_train, y_train, train_index_list,meta_list):
    delay_print.delay_print("---应用训练集 begin---")
    train_result = predicate(clf, x_train, y_train)

    report.predicate_result.append(["id","y_true","y_pred","train"])

    for index, value in enumerate(train_result):
        report.predicate_result.append(
            [
                meta_list[train_index_list[index]][0].get("value"),
                y_train[index],
                value
            ]
        )
        if value != y_train[index]:
            report.train_false_list.append(train_index_list[index])
        else:
            report.train_true_list.append(train_index_list[index])

    delay_print.delay_print("预测正确数量",len(report.train_true_list))
    delay_print.delay_print("预测错误数量",len(report.train_false_list))
    delay_print.delay_print("预测正确数量/预测错误数量",len(report.train_true_list)/len(report.train_false_list))



    train_recall0, train_recall1, train_metrics_confusion_matrix = delay_print_predicate_result(y_train, train_result)
    report.get_cursor().train_recall_0 = train_recall0
    report.get_cursor().train_recall_1 = train_recall1
    report.get_cursor().set_train_score(clf.predict_proba(x_train),y_train
                                        ,map(lambda x:data.meta_x[x][0].get("value"),train_index_list)
                                        ,train_result)
    report.get_cursor().set_train_metrix(train_metrics_confusion_matrix)

    pvalue = Pvalue(clf=clf,y_pred=np.array(train_result),y_true=y_train,x=x_train
                    ,names=list(map(lambda x:x.name,data.description))
                    )
    pvalue_data_frame = pvalue.get_data_frame()
    report.get_cursor().set_train_pvalue_data_frame(pvalue_data_frame)

    delay_print.delay_print("---应用训练集 end---")

def draw_pr_sklearn(y_train_true, y_train_score, y_test_true, y_test_score, filename):
    # Plot PR curve
    plt.figure()
    if y_test_true is not None and y_test_score is not None:
        average_precision = average_precision_score(y_test, y_score)
        precision_test, recall_test, _test = precision_recall_curve(y_test_true, y_test_score)
        

    if y_train_true is not None and y_train_score is not None:
        precision_train, recall_train, _train = precision_recall_curve(y_train_true, y_train_score)





    plt.figure(figsize=(10,6))
    plt.step(recall_test, precision_test, color='b', alpha=0.2,
             where='post')

    plt.plot(recall_train,precision_train,label="train (AUC: {:.3f})".format(auc_train),linewidth=2)
    plt.xlabel("Recall",fontsize=16)
    plt.ylabel("Precision",fontsize=16)
    plt.title(filename + '_Precision Recall Curve')
    plt.legend(fontsize=16)
    plt.savefig(filename + "_pr.png")



def draw_pr(y_train_true, y_train_score, y_test_true, y_test_score, filename):
    # Plot PR curve
    plt.figure()
    if y_test_true is not None and y_test_score is not None:
        precision_test,recall_test,auc_test = CurveLib.get_pr(y_test_score,np.array(y_test_true))

    if y_train_true is not None and y_train_score is not None:
        precision_train,recall_train,auc_train = CurveLib.get_pr(y_train_score,np.array(y_train_true))


    plt.figure(figsize=(10,6))
    plt.plot(recall_test,precision_test,label="test (AUC: {:.3f})".format(auc_test),linewidth=2)
    plt.plot(recall_train,precision_train,label="train (AUC: {:.3f})".format(auc_train),linewidth=2)
    plt.xlabel("Recall",fontsize=16)
    plt.ylabel("Precision",fontsize=16)
    plt.title(filename + '_Precision Recall Curve')
    plt.legend(fontsize=16)
    plt.savefig(filename + "_pr.png")


def draw_roc(y_train_true, y_train_score, y_test_true, y_test_score, filename):
    # Plot ROC curve
    plt.figure()
    if y_test_true is not None and y_test_score is not None:
        fpr_test, tpr_test, thresholds = roc_curve(y_test_true, y_test_score)
        roc_auc_test = auc(fpr_test, tpr_test)
        delay_print.delay_print("fpr_test",fpr_test)
        delay_print.delay_print("tpr_test",tpr_test)
        delay_print.delay_print("thresholds",thresholds)
        delay_print.delay_print("roc_auc_test",roc_auc_test)
        plt.plot(fpr_test, tpr_test, label='ROC test curve (area = %0.3f)' % roc_auc_test,color="red")
    if y_train_true is not None and y_train_score is not None:
        fpr_train, tpr_train, thresholds = roc_curve(y_train_true, y_train_score)
        roc_auc_train = auc(fpr_train, tpr_train)
        delay_print.delay_print("fpr_train",fpr_train)
        delay_print.delay_print("tpr_train",tpr_train)
        delay_print.delay_print("thresholds",thresholds)
        delay_print.delay_print("roc_auc_train",roc_auc_train)
        plt.plot(fpr_train, tpr_train, label='ROC train curve (area = %0.3f)' % roc_auc_train,color="green")

    # spe_x = [0.0614769345238095,
    #          0.0556714923403738,
    #          0.0497937814271455,
    #          0.0438424438822404,
    #          0.0378160875700828,
    #          0.0317132850357434,
    #          0.0255325723776967,
    #          0.0192724480771287,
    #          0.0129313717818296,
    #          0.00650776304260074,
    #          0,
    #          ]
    # sen_y = [0.641025641025641,
    #          0.71099970510174,
    #          0.758144126357354,
    #          0.792064502440059,
    #          0.817640491254187,
    #          0.837613918806959,
    #          0.853643966547192,
    #          0.866793529971456,
    #          0.877775006235969,
    #          0.887083765410762,
    #          0.895074946466809,
    #          ]
    #
    # plt.plot(spe_x, sen_y, label='ROC spe sen',color="blue")

    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title(filename + '_Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename + "_roc.png")


def predicate_test_data(clf, x_test, y_test, test_index_list,meta_list):
    delay_print.delay_print("---应用测试集 begin---")
    test_result = predicate(clf, x_test, y_test)

    report.predicate_result.append(["id","y_true","y_pred","test for ({0}) values".format(len(test_index_list))])
    report_test_false_list = []
    report_test_true_list = []
    for index, value in enumerate(test_result):
        report.predicate_result.append(
            [
                meta_list[test_index_list[index]][0].get("value"),
                y_test[index],
                value
            ]
        )
        if value != y_test[index]:
            report_test_false_list.append(test_index_list[index])
        else:
            report_test_true_list.append(test_index_list[index])

    delay_print.delay_print("预测正确数量",len(report_test_true_list))
    delay_print.delay_print("预测错误数量",len(report_test_false_list))
    delay_print.delay_print("预测正确数量/预测错误数量",len(report_test_true_list)/len(report_test_false_list))
    report.test_false_list = report.test_false_list + report_test_false_list
    report.test_true_list = report.test_true_list + report_test_true_list
    test_recall0, test_recall1, test_metrics_confusion_matrix = delay_print_predicate_result(y_test, test_result)
    report.get_cursor().test_recall_0 = test_recall0
    report.get_cursor().test_recall_1 = test_recall1
    report.get_cursor().set_test_metrix(test_metrics_confusion_matrix)
    report.get_cursor().set_test_score(clf.predict_proba(x_test),y_test
                                       ,map(lambda x:data.meta_x[x][0].get("value"),test_index_list)
                                       ,test_result)

    delay_print.delay_print("---应用测试集 end---")


def train_test_split_with_index(x, y, test_size):
    _x = [value + [index] for index, value in enumerate(x)]
    x_train, x_test, y_train, y_test = train_test_split(_x, y, test_size=test_size)

    return x_train, x_test, y_train, y_test, [i.pop() for i in x_train], [i.pop() for i in x_test]





def run(weight, clf_func, time):
    global delay_print
    delay_print = DelayPrint()
    delay_print.delay_print("target_is",target_is, "数据分为两组, 是",target_is,"的(新模型的y1)和不是",target_is,"的(新模型的y0)")
    report.new()
    # clf = LogisticRegression(class_weight='balanced')
    # clf = RandomForestClassifier(max_depth =3,class_weight='balanced')
    # clf = SVC(class_weight={0:1,1:weight})
    # clf = DecisionTreeClassifier(class_weight='balanced')


    class_weight = 'balanced' if weight is None else {0: 1, 1: weight}
    delay_print.delay_print("-----------------一次训练 begin-----------------")
    delay_print.add_print_task(adapter.print_metadata)
    delay_print.delay_print("0:权重,1:权重", class_weight)
    #可以为报告添加内容
    report.get_cursor().class_weight = class_weight
    clf = clf_func(class_weight)

    x_train, x_test, y_train, y_test, train_index_list, test_index_list = [None,None,None,None,None,None]

    if g_test_size is not None:

        x_train, x_test, y_train, y_test, train_index_list, test_index_list = train_test_split_with_index(data.x, data.y,
                                                                                                          test_size=0.5)
    else:
        x_train, y_train, train_index_list = data.x, data.y,range(len(data.x))

    fit(clf, x_train, y_train, data.description)

    delay_print.delay_print("训练方法是: ", type(clf))
    delay_print.delay_print("训练方法详细参数是: ", str(clf))
    report.get_cursor().algorithm = type(clf)


    predicate_train_data(clf, x_train, y_train, train_index_list,data.meta_x)

    if g_test_size is not None:
        predicate_test_data(clf, x_test, y_test, test_index_list,data.meta_x)
    if data_2 is not None:
        predicate_test_data(clf, data_2.x, data_2.y, range(len(data_2.x)),data_2.meta_x)

    def dynamic_call(funcs,*args):
        for i in funcs:
            i(*args)
    file_name = g_folder+ToolLib.make_name(class_weight,weight, clf, time,target_is)
    if hasattr(clf, "predict_proba"):
        dynamic_call([draw_roc,draw_pr],y_train,
                 clf.predict_proba(x_train)[:, 1],
                 None if y_test is None else  y_test,
                 None if y_test is None else clf.predict_proba(x_test)[:, 1],
                     file_name)

    delay_print.delay_print("-----------------一次训练 end-----------------")
    if not delay_print.will_print:
        report.remove_last()


    #打印所有文件
    delay_print.print_all(file_name)



#剪枝

def post_prune(tree,gini=0.15):
    # gini>0.25 所有节点(包括叶子节点和路径节点) delete
    # pure_y_0_count>500 叶子节点 同级节点指定深度 delete
    tree_post_prune_func = tree_post_prune(tree)
    tree_post_prune_func.assess_gini()
    tree_post_prune_func.post_prune_gini(gini)

    return





excel_data_source_path = "D:\\info_data\\jingzong\\补全缺失值数据分析.xlsx"

#excel_data_source_path = "E:\\info_data\\国妇婴项目_20180419_完整数据_数据和变量类型有改动.xlsx"


def get_column_variable_type_mapping():
    name_type_mapping = {}
    name_categorical_reduce_mapping = {}
    xl = pd.ExcelFile(excel_data_source_path)
    meta_frame = xl.parse("描述分析")

    for indexs in meta_frame.index:
        if meta_frame.loc[indexs]["方法-1"] == "Y":
            name = meta_frame.loc[indexs]["变量名称"]
            type = ResearchDataRawColumnType.categorical if meta_frame.loc[indexs]["变量类型"] == "字符串" else ResearchDataRawColumnType.numerical
            name_type_mapping[name] = type
            if type == ResearchDataRawColumnType.categorical:
                categorical_reduce = meta_frame.loc[indexs]["对照值（即减去的值）"]
                if categorical_reduce != categorical_reduce:# only nan != nan

                    raise ValueError("if it is categorical, it must provide a reduce name")
                name_categorical_reduce_mapping[name] = categorical_reduce
    return name_type_mapping,name_categorical_reduce_mapping
name_type_mapping,name_categorical_reduce_mapping = get_column_variable_type_mapping()


print(name_type_mapping)
print(name_categorical_reduce_mapping)

#*******
adapter = ExcelAdapter(
    path=excel_data_source_path,
    #sheet_name="11025"
    sheet_name="data"
    , result_index="PSP三分类"
    ,ignore_column_name_list=["患者编号","PSP总评分"]
    , meta_column_name_list=["No"]
    , valid_column_name_list= [key for key in name_type_mapping]
    , column_name_type_mapping_dict=name_type_mapping
    , column_name_categorical_reduce_mapping = name_categorical_reduce_mapping  #onehot去掉一项
)
# ,"weight_before_pregnant","'height","height"



#*****
data = adapter.fill_to_new()


data_2 = None
# data_2 = ExcelAdapter(
#     path=excel_data_source_path,
#     #sheet_name="11025"
#     sheet_name="test"
#     , result_index="新生儿是否感染"
#     ,valid_column_name_list=valid_column_name_list
#     , meta_column_name_list=["id"]
#     , column_name_type_mapping_dict=get_column_variable_type_mapping()
# ).fill_to_new()

delay_print = None
report = None

def init_context():
    global delay_print
    global report
    delay_print = None
    report = Report()
    report.meta_x = data.meta_x
    report.excel = adapter

def empty_method(*args):
    return None
g_test_size = None
g_test_size = 0.3
g_cut_off_lambda = lambda x:1 if x[1]>=0.4 else 0
Report.save_calculate_mean = empty_method

Report.save_p_value_data_frame = empty_method
Report.save_proba = empty_method
g_cut_off_lambda = lambda x:1 if x[1]>=0.5 else 0
Pvalue.get_data_frame = empty_method





random_func = lambda class_weight: RandomForestClassifier(
    max_depth=4,
    # ,min_samples_split=10
    class_weight=class_weight
    ,n_estimators=1000
)


logi_func = lambda class_weight: LogisticRegression(class_weight=class_weight,C=0.99, penalty='l1')
custom_logi_func = lambda class_weight: CustomLinearRegression(class_weight=class_weight)
lr_bagging = lambda class_weight: BaggingRegressor( n_estimators=10
                                                    ,base_estimator=LogisticRegression(class_weight=class_weight)
                                                    )

svm_func = lambda class_weight:SVC(class_weight=class_weight,probability=True)
dc_func = lambda class_weight:sklearn.tree.DecisionTreeClassifier()

gbdt_func = lambda class_weight:GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                           min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10
                                                           ,n_estimators = 40)

# init_context()
# run(None,logi_func,0)
# report.save()


target_is = 0


data.y = list(map(lambda x:1 if int(x) == target_is else 0 ,data.y))
sgd_func = lambda class_weight:SGDClassifier(class_weight=class_weight,loss="log", penalty="elasticnet",alpha=0.0001)

g_folder="output/"+adapter.sheet_name+"_"+ToolLib.get_safe_date_time_now()+"/"
import os
if not os.path.exists(g_folder):
    os.makedirs(g_folder)



gsearch = GridSearchCV( LogisticRegression(class_weight="balanced") , {
    "C":np.arange(0.01,0.7,0.01),
    "penalty":['l1','l2']
}, scoring='roc_auc', cv=5 )

gsearch.fit( data.x, data.y )
print(gsearch.best_estimator_)
print(gsearch.best_params_)
print(gsearch.best_score_)
result = sorted(
    [{'x': data.description[index].name, 'rank': item} for index, item in enumerate(gsearch.best_estimator_.coef_[0])],
    key=lambda x: x["rank"], reverse=True)
print(result)

init_context()


run(None,random_func,0)
# None : balance Number: 0:1,1:Number -> class_weight
# random_func: method
# 0 label, None: use timestamp instead
run(None,logi_func,1)
run(None,logi_func,2)
run(None,logi_func,3)
run(None,logi_func,4)
report.save(pre_fix=g_folder)



# Linq.range(1, 1, lambda true_weight: Linq.range(0, 1, lambda time: run(true_weight
#                                                                          #,dc_func
#                                                                          #,logi_func
#                                                                          #,lr_bagging
#                                                                          , random_func_balance
#                                                                          #,svm_func
#                                                                          , time)))

# Linq.range(1, 1, lambda true_weight: Linq.range(0, 1, lambda time: run_whole_dc_tree(true_weight
#                                                                                        ,dc_func
#                                                                                        #,logi_func
#                                                                                        #,lr_bagging
#                                                                                        #, random_func
#                                                                                        #,svm_func
#                                                                                        , time)))




