from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, classification_report, average_precision_score, roc_auc_score
#可以计算roc pr 混洗矩阵
def get_roc_pr(y_test, pre_test,pre_test_label):
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pre_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, lw=2, alpha=0.8, color="r",
             label='(test AUC = %0.3f)' % (roc_auc_test))
    plt.legend(loc="lower right")
    plt.xlabel('1-Specificity,%')
    plt.ylabel('Sencitivity,%')
    plt.figure()

    print(classification_report(y_test, pre_test_label))
    precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, pre_test)
    AP_test = average_precision_score(y_test, pre_test)

    plt.plot(recall_test, precision_test, lw=2, alpha=0.8, color="b",
             label='(test AP = %0.3f)' % (AP_test))
    plt.legend(loc="lower left")
    plt.xlabel('Recall,%')
    plt.ylabel('Precision,%')
    plt.show()
