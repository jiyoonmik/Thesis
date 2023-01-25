from cgi import test
import pickle
from pickle import dump
from pickle import HIGHEST_PROTOCOL
from this import d
import pandas as pd

import datetime
from datetime import time
import os
import re
from collections import Counter

import numpy as np
import math
import csv


def p_by_d(folder,id): #patient_by_diagnosis
    #patient에 대한 diagnosis label
    load_diagnosis = pd.read_excel(folder+'/data/diagnosis.xlsx', index_col='ID', engine='openpyxl')
    dd=load_diagnosis['Diagnosis'][id]
    return dd

def open_datafile(folder, id):
    NAdata={}
    filename = str(id)
    if len(filename) == 1:
        filename='00'+filename
    elif len(filename) == 2:
        filename='0'+filename
    with open(folder+"/data/{}.txt".format(str(filename)), 'r') as features:
        rawdata = features.readlines()
        if len(rawdata) == 1:
            NAdata[filename]=rawdata
            rawdata = 'U'
            print(NAdata, rawdata)
    return rawdata


def add_dict(dictSensors, dictObs, new_sensor):
    if new_sensor not in dictSensors.keys():
        new_sensor_val = max(dictSensors.values())+1
        dictSensors[new_sensor] = new_sensor_val

        new_ob_val = max(dictObs.values())+1
        if ("M" or "AD") in new_sensor:
            dictObs[new_sensor + "OFF"] = new_ob_val 
            dictObs[new_sensor + "ON"] = new_ob_val+1
        elif "D" in new_sensor:
            dictObs[new_sensor + "CLOSE"] = new_ob_val
            dictObs[new_sensor + "OPEN"] = new_ob_val+1
        elif "I" in new_sensor:
            dictObs[new_sensor + "ABSENT"] = new_ob_val
            dictObs[new_sensor + "PRESENT"] = new_ob_val+1

    return dictSensors, dictObs

#make a patient data into feature 
def make_feature(id,rawdata, dictSensors, dictObs):
    IndexErrorlist=[]

    timestamps = []
    sensors = []
    values = []
    task = []
    subactivity = []
    labeled_only = []
    non_0 = []
    sensorvalues = []

    for i, line in enumerate(rawdata):  # each line
        f_info = line.split()  # find fields
        try:
            if (len(f_info)>=3) and (('M' == f_info[1][:-3]) or ('D' == f_info[1][:-3]) or ('AD' == f_info[1][:-3]) or ('I' == f_info[1][:-3])):

                if not ('.' in str(np.array(f_info[0]))):
                    f_info[0] = f_info[0] + '.000000'

                if len(f_info)== 3:  # if activity does not exist append '0'
                    task.append(0)
                    subactivity.append(0)
                    non_0.append(0)
                    write = 2

                elif len(f_info) == 4:  # if activity exists
                    des = str(' '.join(np.array(f_info[3:]))) #activity label
                    split_label = des.split(',')
                    write = len(split_label)+1

                    for v, l in enumerate(split_label):
                        
                        if 'start' in l:
                            Act = int(re.sub('-start|start', '', l))
                            task.append(Act)
                            subactivity.append(Act+0.001)#'<s>{}'.format(Act))
                            labeled_only.append(Act+0.001)
                            non_0.append(1)

                        elif 'end' in l:
                            Act = int(re.sub('-end|end', '', l))
                            task.append(Act)
                            subactivity.append(Act+0.009)#'<e>{}'.format(Act))
                            labeled_only.append(Act+0.009)
                            non_0.append(1)

                        elif 'incomplete' in l:
                            index = re.sub('-incomplete|incomplete', '', l)
                            if index == '':
                                index = re.sub('-start|start', '', split_label[v-1])
                            Act = int(index)
                            task.append(Act)
                            subactivity.append(Act+0.004)#'<i>{}'.format(Act))
                            labeled_only.append(Act+0.004)
                            non_0.append(1)

                        else:
                            #print(id,f_info, l)
                            if '.' in l:
                                Act, sub_label =l.split('.') #breakdown a label into Act+sub_label
                                Act, sub_label = int(Act), int(sub_label)
                                task.append(Act)
                                subactivity.append(float('{}.{}'.format(Act, sub_label)))
                                labeled_only.append(float('{}.{}'.format(Act, sub_label)))
                                non_0.append(1)
                            elif int(l) > 16:
                                task.append(int(l))
                                subactivity.append(float(l))
                                labeled_only.append(float(l))
                                non_0.append(1)
                            else:
                                #activities.append(0)
                                print("GOT YA!!!!!!!!!!!!!!!!!!!!!",i,split_label,v,l)

                dictSensors, dictObs = add_dict(dictSensors, dictObs, f_info[1])

                for w in range(1, write): # 라벨 복제한 만큼 다른 feature도 복제 append
                    timestamps.append(f_info[0])
                    sensors.append(f_info[1])
                    if 'OF' in f_info[2]:
                        values.append('OFF')
                        sensorvalues.append(str(f_info[1])+'OFF')
                    elif 'ON' in f_info[2]:
                        values.append('ON')
                        sensorvalues.append(str(f_info[1])+'ON')
                    else:
                        values.append(f_info[2])
                        sensorvalues.append(str(f_info[1])+str(f_info[2]))
                
        except IndexError as e:
            IndexErrorlist.append([i, line, e])

    if len(IndexErrorlist)>0:
        print('ID:',id,'IndexErrorlist:',IndexErrorlist)
        #print(len(timestamps), len(sensors), len(values),len(activities))

    task=np.array(task, dtype=np.float64)
    subactivity=np.array(subactivity, dtype=np.float64)
    labeled_only=np.array(labeled_only, dtype=np.float64)
    non_0=np.array(non_0, dtype=np.float64)

    return dictSensors, dictObs, timestamps, sensors, values, sensorvalues, non_0, task, labeled_only, subactivity

def get_time(time):
    """ Convert a pair of date and time strings to a datetime structure.
    """
    try:
        dt = datetime.datetime.strptime(time, "%H:%M:%S.%f")
    except:
        dt = datetime.datetime.strptime(time, "%H:%M:%S")
    return dt

def featurize(timestamps, sensors, values, dictObs):
 
    sv = [] #sensor+value 고유값으로 표현된 시퀀스
    timegap = [] #이전 센서 활성화 시간 간격

    for kk, s in enumerate(sensors): #kk<=전체 시퀀스에서의 순서
        state=str(values[kk])
        #print(kk, s, state, id)
        sv.append(dictObs[s + state])

    for kk, t in enumerate(timestamps):
        current=get_time(t)
        if kk==0:
            tg=0
        else:
            prior=get_time(timestamps[kk-1])
            tg= (current-prior).total_seconds()
        timegap.append(tg)

    sv=np.array(sv, dtype=np.float64)
    timegap=np.array(timegap, dtype=np.float64)

    return sv, timegap

#main
folder='/home/jiyoon/OneDrive/Projects/Thesis/assessmentdata/assessmentdata'

df=pd.DataFrame(columns=['id', 'timestamps', 'sensors', 'values', 'sensorvalues', 'timegap', 'sv', 'non_0', 'task', 'labeled_only', 'subactivity', 'diagnosis'])

dictSensors={'D007': 0}
dictObs={'D007CLOSE': 0, 'D007OPEN': 1}
for id in range(1, 401):
    #print('**********start',id,'**********')
    rawdata = open_datafile(folder, id)
    if rawdata == 'U':
        continue
    dictSensors, dictObs, timestamps, sensors, values, sensorvalues, non_0, task, labeled_only, subactivity = make_feature(id,rawdata, dictSensors, dictObs)
    sv, timegap = featurize(timestamps, sensors, values, dictObs)
    #print(len(sv))
    if (len(sv)==len(timestamps)==len(sensors)==len(non_0)==len(task)==len(subactivity))==False:
        print('******************',id,len(sv),len(timestamps),len(sensors),len(non_0),len(task),len(subactivity))
    diagnosis=p_by_d(folder,id)
    df.loc['{}'.format(id)]=[id, timestamps, sensors, values, sensorvalues, timegap, sv, non_0, task, labeled_only, subactivity, diagnosis]


print(df.iloc[0:3,:])
df.info()
df1=df.set_index('id')
df1.isna().sum()
df2=df1.dropna()

df2['redefined'] = df2.apply(lambda x : 0 if ((x['diagnosis'] != 1) or (x['diagnosis']!= 2)) else 1, axis = 1)

g1=[1,2,7] #104
g2=[4,5] #120
g3=[3,8,9] #108
df2['group'] = df2.apply(lambda x : 1 if x['diagnosis'] in g1 else (2 if (x['diagnosis'] in g2) else (3 if (x['diagnosis'] in g3) else np.nan)), axis = 1)
#dimentia or mci 아닐 경우 0
df2.info()
df3 = pd.DataFrame.dropna(df2)
df3.info()
print(df3.iloc[0:3,:])

df3.to_pickle('{}/data.pkl'.format(folder))
dfpickle=pd.read_pickle('{}/data.pkl'.format(folder))
dfpickle

with open('{}/newdictObs.pkl'.format(folder), 'wb') as p:
    pickle.dump(dictObs, p, protocol=pickle.HIGHEST_PROTOCOL)


