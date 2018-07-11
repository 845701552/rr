import pandas as pd
import numpy as np
import time
import datetime as dt
data=pd.read_excel("C:/Users/gang.yu/Desktop/关联宫腔分泌物-2018.xlsx")#parse_dates=["医嘱具体时间","胎儿娩出具体时间"]
print(data[data["医嘱时间"].isnull()])
data=data[data["医嘱时间"].notnull()]

def wanqian(data):
    data=data[data["何时检查"]=="娩出前检查"]
    data["胎儿娩出日期"]=data["胎儿娩出日期"].astype(str)
    data["胎儿娩出具体时间"]=data["胎儿娩出具体时间"].astype(str)
    data["医嘱日期"]=data["医嘱日期"].astype(str)
    data["医嘱具体时间"]=data["医嘱具体时间"].astype(str)
    data["医嘱具体时间_日期"]=data["医嘱日期"]+" "+data["医嘱具体时间"]
    data["胎儿具体时间_日期"]=data["胎儿娩出日期"]+" "+data["胎儿娩出具体时间"]
    data["胎儿娩出时间_日期_date"]=[dt.datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in data["胎儿具体时间_日期"]]
    data["医嘱具体时间_日期_date"]=[dt.datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in data["医嘱具体时间_日期"]]
    data["diff"]=data["医嘱具体时间_日期_date"]-data["胎儿娩出时间_日期_date"]
    data["abs_diff_hour"]=[ np.abs(i.total_seconds()/3600) for i in data["diff"]]
    data["diff_hh_2wei"]=[str(round(i,2)) for i in data["abs_diff_hour"]]
    data_group=data.groupby(["name1","name"])["abs_diff_hour"].apply(lambda x:str(round(x.min(),2)))
    data_group.to_csv("C:/Users/gang.yu/Desktop/关联宫腔分泌物-2018.csv",encoding="utf_8_sig")
#wanqian(data)



def finall_result(data):
    data["胎儿娩出日期"] = data["胎儿娩出日期"].astype(str)
    data["胎儿娩出具体时间"] = data["胎儿娩出具体时间"].astype(str)
    data["医嘱日期"] = data["医嘱日期"].astype(str)
    data["医嘱具体时间"] = data["医嘱具体时间"].astype(str)
    data["医嘱具体时间_日期"] = data["医嘱日期"] + " " + data["医嘱具体时间"]
    data["胎儿具体时间_日期"] = data["胎儿娩出日期"] + " " + data["胎儿娩出具体时间"]
    data["胎儿娩出时间_日期_date"] = [dt.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in data["胎儿具体时间_日期"]]
    data["医嘱具体时间_日期_date"] = [dt.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in data["医嘱具体时间_日期"]]
    data["diff"] = data["医嘱具体时间_日期_date"]-data["胎儿娩出时间_日期_date"]
    data["abs_diff_hour"] = [np.abs(i.total_seconds()/3600) for i in data["diff"]]
    data["diff_hh_2wei"] = [str(round(i, 2)) for i in data["abs_diff_hour"]]

    data2=pd.read_excel("C:/Users/gang.yu/Desktop/关联宫腔分泌物-2018.xlsx",sheet_name="new")
    data2["diff_hh_2wei"]=[str(round(i,2)) for i in data2["diff_hh_2wei"]]
    data3=pd.read_excel("C:/Users/gang.yu/Desktop/关联宫腔分泌物-2018.xlsx",sheet_name="new_wanqian")
    data3["wanqian_diff_hh_2wei"]=[str(round(i,2)) for i in data3["wanqian_diff_hh_2wei"]]
    data4=pd.merge(left=data2,right=data3,on=["name1","name"],how="left")
    data4["diff_hh_2wei"]=data4["diff_hh_2wei"].astype(float)
    data4["wanqian_diff_hh_2wei"]=data4["wanqian_diff_hh_2wei"].astype(float)
    data4.loc[data4["diff_hh_2wei"]<=12,"finall_diff"]=data4["diff_hh_2wei"]
    data4.loc[((data4["diff_hh_2wei"]>12) & (data4["wanqian_diff_hh_2wei"].notnull())),"finall_diff"]=data4["wanqian_diff_hh_2wei"]
    data4.loc[((data4["diff_hh_2wei"]>12) & (data4["wanqian_diff_hh_2wei"].isnull())),"finall_diff"]=data4["diff_hh_2wei"]
    data4["diff_hh_2wei"]=[ str(round(i,2)) for i in  data4["finall_diff"]]
    data5=pd.merge(left=data,right=data4,on=["name1","name","diff_hh_2wei"])[["name1","门诊号","name","报告值","何时检查","diff","abs_diff_hour"]]
    data5.to_csv("C:/Users/gang.yu/Desktop/关联宫腔分泌物-2018ok.csv",encoding="utf_8_sig")
finall_result(data)




