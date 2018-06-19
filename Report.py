
from itertools import groupby

import functools

import numpy as np
import pandas as pd
import xlrd
import xlutils
import xlwt as xlwt

from sklearn_ml_core.Linq.Linq import Linq
from xlutils.copy import copy


class Report:
    def __init__(self):
        self.data=[]
        self.train_true_list=[]
        self.train_false_list=[]
        self.test_true_list=[]
        self.test_false_list=[]
        self.train_meta_data=[]
        self.predicate_result=[]
        self.wbk=None
        self.meta_x=None
        self.excel=None

    def group_out(self,_list,sheet,id_column,count_column,row=1):
        groups = groupby(sorted(_list))
        grouped_dict = [{'item':k, 'count':len(list(v))} for k, v in groups]
        for index,value in enumerate(sorted(grouped_dict, key=lambda tup: tup["count"],reverse=True)):
            sheet.write(index+row,id_column,str(self.meta_x[value["item"]]))
            sheet.write(index+row,count_column,value["count"])


    def new(self):
        self.data.append(ReportData())

    def get_algorithm_list_name(self):
        algorithm_list_name = functools.reduce(lambda current,next:current+"_"+next,set(map(lambda x:x.algorithm.__name__,self.data)))
        return algorithm_list_name

    def get_cursor(self):
        return self.data[len(self.data)-1]

    def remove_last(self):
        return self.data.pop()

    def save_predicate_result(self):
        if self.predicate_result is None:
            return
        sheet = self.wbk.add_sheet('predicate_compare',cell_overwrite_ok=True)
        self.put_2_d_array_in_sheet(self.predicate_result,sheet)


    def save_stats(self):
        sheet = self.wbk.add_sheet('y_stats',cell_overwrite_ok=True)

        sheet.write(0,0,"train_true_list_id")
        sheet.write(0,1,"train_true_list_count")
        sheet.write(0,2,"train_false_list_id")
        sheet.write(0,3,"train_false_list_count")
        sheet.write(0,4,"test_true_list_id")
        sheet.write(0,5,"test_true_list_count")
        sheet.write(0,6,"test_false_list_id")
        sheet.write(0,7,"test_false_list_count")

        self.group_out(self.train_true_list,sheet,0,1)
        self.group_out(self.train_false_list,sheet,2,3)
        self.group_out(self.test_true_list,sheet,4,5)
        self.group_out(self.test_false_list,sheet,6,7)

    def append_remark(self,sheet,row,last_column_index,remark):
        if Linq.is_null_or_empty(remark):
            return
        for index,item in enumerate(remark):
            sheet.write(row,last_column_index+index,str(item))
        return

    def save_list(self):
        sheet = self.wbk.add_sheet('list',cell_overwrite_ok=True)

        dict_keys = list(self.data[0].__dict__.keys())

        for j,value in enumerate(dict_keys):
            sheet.write(0,j,value)

        for i,row in enumerate(self.data):
            for j,value in enumerate(list(row.__dict__.values())):
                sheet.write(1+i,j,str(value))
            self.append_remark(sheet,1+i,j,row.remark)

    def save_meta(self):
        return

    def put_2_d_array_in_sheet(self,two_d_array,sheet):
        for row_index,row in enumerate(two_d_array):
            for column_index,value in enumerate(row):
                sheet.write(row_index,column_index,str(value))


    def save_calculate_mean(self):
        sheet = self.wbk.add_sheet('mean',cell_overwrite_ok=True)

        sheet_array=[]
        y_array=["weight","mean_train_recall_0","mean_train_recall_1","mean_test_recall_0","mean_test_recall_1"
            ,"weight","mean_train_precision_0","mean_train_precision_1","mean_test_precision_0","mean_test_precision_1"]
        sheet_array.append(y_array)
        sorted_input = sorted(self.data, key=lambda obj:str(obj.class_weight))
        groups = groupby(sorted_input, key=lambda obj:str(obj.class_weight))
        for k, v in groups:
            v = list(v)
            v_precision_list = list(map(lambda x:x.get_precision_list(),v))

            _array = [k,np.mean(list(map(lambda x:x.train_recall_0,v)))
                ,np.mean(list(map(lambda x:x.train_recall_1,v)))
                ,np.mean(list(map(lambda x:x.test_recall_0,v)))
                ,np.mean(list(map(lambda x:x.test_recall_1,v)))
                      ,k
                ,np.mean(list(map(lambda x:x.get("train_precision_0"),v_precision_list)))
                ,np.mean(list(map(lambda x:x.get("train_precision_1"),v_precision_list)))
                ,np.mean(list(map(lambda x:x.get("test_precision_0"),v_precision_list)))
                ,np.mean(list(map(lambda x:x.get("test_precision_1"),v_precision_list)))]
            sheet_array.append(_array)
        self.put_2_d_array_in_sheet(sheet_array,sheet)

    def save_proba(self):
        for i in self.data:
            sheet = self.wbk.add_sheet('proba_'+str(i.class_weight).replace(":","_"),cell_overwrite_ok=True)
            self.put_2_d_array_in_sheet(i.get_score_2d_list(),sheet)

    def save_p_value_data_frame(self,path):

        writer = pd.ExcelWriter(path)

        for i in self.data:
            if i.train_pvalue_data_frame is not None:
                sheet_name = 'p_value_'+str(i.class_weight).replace(":","_")
                i.train_pvalue_data_frame.to_excel(writer, sheet_name=sheet_name)

        writer.save()


    def get_name(self):
        return self.excel.sheet_name+"_"+self.get_algorithm_list_name()+"_"+self.excel.filename

    def save(self,pre_fix="",post_fix=""):
        if len(self.data) == 0:
            return

        path = pre_fix+self.get_name()+post_fix+".xls"

        self.wbk = xlwt.Workbook()

        self.wbk.add_sheet('empty',cell_overwrite_ok=True)
        self.wbk.save(path)
        self.save_p_value_data_frame(path)

        self.wbk = copy( xlrd.open_workbook(path))


        self.save_predicate_result()
        self.save_list()
        self.save_stats()
        self.save_calculate_mean()
        self.save_proba()

        self.wbk.save(path)

class Proba:
    def __init__(self,y_list,id_index_list,proba_ndarray,y_pred_list):
        self.y_list = y_list
        self.id_index_list = id_index_list
        self.proba_ndarray = proba_ndarray
        self.y_pred_list = y_pred_list

    def to_2_d_array(self):
        result = []
        title = ["id","y_true","y_pred","0","1"]
        result.append(title)
        for y_item,score_item,index,y_pred_item in zip(self.y_list,self.proba_ndarray.tolist(),self.id_index_list,self.y_pred_list):
            result.append([
                index,
                y_item,
                y_pred_item,
                score_item[0],
                score_item[1]
            ])
        return result



class ReportData:
    def __init__(self):
        self.msg=None
        self.feature_rank=None
        self.class_weight=None
        self.algorithm=None
        self.algorithm_full=None
        self.train_recall_0=None
        self.train_recall_1=None
        self.train_tn=None
        self.train_fn=None
        self.train_fp=None
        self.train_tp=None
        self.train_acc=None
        self.test_recall_0=None
        self.test_recall_1=None
        self.test_tn=None
        self.test_fn=None
        self.test_fp=None
        self.test_tp=None
        self.test_acc=None
        self.train_score=None
        self.test_score=None
        self.train_pvalue_data_frame=None
        self.remark=[]

    def get_score_2d_list(self):
        return self.train_score.to_2_d_array()+\
               [] if self.test_score is None else self.test_score.to_2_d_array()

    def set_train_score(self,score,y_list,id_list,y_pred_list):
        self.train_score = Proba(y_list,id_list,score,y_pred_list)

    def set_test_score(self,score,y_list,id_list,y_pred_list):
        self.test_score = Proba(y_list,id_list,score,y_pred_list)

    def set_train_pvalue_data_frame(self,train_pvalue_data_frame):
        self.train_pvalue_data_frame = train_pvalue_data_frame

    def set_train_metrix(self,matrix):
        self.train_tn=matrix[0][0]
        self.train_fp=matrix[0][1]
        self.train_fn=matrix[1][0]
        self.train_tp=matrix[1][1]
        self.train_acc=(self.train_tn+self.train_tp)/(self.train_tn+self.train_tp+self.train_fn+self.train_fp)


    def set_test_metrix(self,matrix):
        self.test_tn=matrix[0][0]
        self.test_fp=matrix[0][1]
        self.test_fn=matrix[1][0]
        self.test_tp=matrix[1][1]
        self.test_acc=(self.test_tn+self.test_tp)/(self.test_tn+self.test_tp+self.test_fn+self.test_fp)

    def get_precision_list(self):
        return {
            "train_precision_0":self.train_tn/(self.train_tn+self.train_fn),
            "train_precision_1":self.train_tp/(self.train_tp+self.train_fp),
            "test_precision_0":self.test_tn/(self.test_tn+self.test_fn),
            "test_precision_1":self.test_tp/(self.test_tp+self.test_fp)
        }