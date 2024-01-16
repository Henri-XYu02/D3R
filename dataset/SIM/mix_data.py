import os
import numpy as np
import pandas as pd
import pickle as pk
import random
random.seed(0)

pth = './'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_folder = 'data/processed_sim_data'
columns = ['agent_id', 'latitude', 'longitude', 'time', 'stay_minutes']
baseline = pd.DataFrame(np.load(os.path.join(ROOT_DIR, 'baseline/data_stays_v2.npy'), allow_pickle=True), columns=columns)
baseline['time'] = np.array([x.timestamp() for x in baseline['time']])
kitware = pd.DataFrame(np.load(os.path.join(ROOT_DIR, 'kitware/data_stays_v2.npy'), allow_pickle=True), columns=columns)
kitware['time'] = np.array([x.timestamp() for x in kitware['time']])
l3harris = pd.DataFrame(np.load(os.path.join(ROOT_DIR, 'l3harris/data_stays_v2.npy'), allow_pickle=True), columns=columns)
l3harris['time'] = np.array([x.timestamp() for x in l3harris['time']])
min_agent = min(baseline['agent_id'].max(), kitware['agent_id'].max(), l3harris['agent_id'].max())

data_dims = 4
b9k1, k9l1, l9b1 = [], [], []
b9k1_tst, k9l1_tst, l9b1_tst = [],[],[]
lab1, lab2, lab3 =[],[],[]
timegap = 1.4 * 86400
ids = list(set(baseline['agent_id']).intersection(kitware['agent_id']).intersection(l3harris['agent_id']))
ids_trn = ids[0:int(0.8*len(ids))]
ids_tst = ids[int(0.8*len(ids)):]

for i in ids_trn:
    t = random.uniform(0, 14-1.4) * 86400
    d1,d2,d3 = baseline[baseline['agent_id'] ==i], kitware[kitware['agent_id']==i], l3harris[l3harris['agent_id']==i]
    t1,t2,t3 = d1.iloc[0]['time'] + t, d2.iloc[0]['time'] + t, d3.iloc[0]['time'] + t
    i1, i2, i3 = d1[(d1['time'] >= t1) & (d1['time'] < t1 + timegap)], d2[(d2['time'] >= t2) & (d2['time'] < t2 + timegap)], d3[(d3['time'] >= t3) & (d3['time'] < t3 + timegap)]
    
    mix1 = pd.concat([d1[(d1['time'] < t1)], i2, d1[(d1['time'] >= t1 + timegap)]], axis=0)
    mix2 = pd.concat([d2[(d2['time'] < t2)], i3, d2[(d2['time'] >= t2 + timegap)]], axis=0)
    mix3 = pd.concat([d3[(d3['time'] < t3)], i1, d3[(d3['time'] >= t3 + timegap)]], axis=0) 
    b9k1.append(np.array(mix1)[:, 1:])
    k9l1.append(np.array(mix2)[:, 1:])
    l9b1.append(np.array(mix3)[:, 1:])

for i in ids_tst:
    t = random.uniform(0, 14-1.4) * 86400
    d1,d2,d3 = baseline[baseline['agent_id']==i], kitware[kitware['agent_id']==i], l3harris[l3harris['agent_id']==i]
    t1,t2,t3 = d1.iloc[0]['time'] + t, d2.iloc[0]['time'] + t, d3.iloc[0]['time'] + t
    i1, i2, i3 = d1[(d1['time'] >= t1) & (d1['time'] < t1 + timegap)], d2[(d2['time'] >= t2) & (d2['time'] < t2 + timegap)], d3[(d3['time'] >= t3) & (d3['time'] < t3 + timegap)]
    
    mix1 = pd.concat([d1[(d1['time'] < t1)], i2, d1[(d1['time'] >= t1 + timegap)]], axis=0)
    mix2 = pd.concat([d2[(d2['time'] < t2)], i3, d2[(d2['time'] >= t2 + timegap)]], axis=0)
    mix3 = pd.concat([d3[(d3['time'] < t3)], i1, d3[(d3['time'] >= t3 + timegap)]], axis=0)

    # some label for one series [0,0,0,1,1,0,0,0]
    # label1 = np.zeros(len(mix1))
    # label1[len(d1[(d1['time'] < t1)]):len(d1[(d1['time'] < t1)])+len(i2)] = 1
    # lab1.append(label1)
    
    # label2 = np.zeros(len(mix2))
    # label2[len(d2[(d2['time'] < t2)]):len(d2[(d2['time'] < t2)])+len(i3)] = 1
    # lab2.append(label2)
    
    # label3 = np.zeros(len(mix3))
    # label3[len(d3[(d3['time'] < t3)]):len(d3[(d3['time'] < t3)])+len(i1)] = 1
    # lab3.append(label3)
    
    # some label for one series: 1 or 0
    for x in ['1','2','3']:
        if len(eval('i'+x))!= 0:
            eval('lab'+x).append(1)
        elif len(eval('d'+x)) != len(eval('mix'+x)):
            eval('lab'+x).append(1)
        else:
            eval('lab'+x).append(0)
    
    b9k1_tst.append(np.array(mix1)[:, 1:])
    k9l1_tst.append(np.array(mix2)[:, 1:])
    l9b1_tst.append(np.array(mix3)[:, 1:])

assert len(b9k1_tst) == len(lab1)
assert len(k9l1_tst) == len(lab2)
assert len(l9b1_tst) == len(lab3)
lab1 = np.array(lab1)
lab2 = np.array(lab2)
lab3 = np.array(lab3)

with open(pth + 'b9k1.pk', 'wb') as file:
    pk.dump({'x_trn': b9k1, 'x_tst': b9k1_tst, 'lab_tst': lab1}, file)

with open(pth + 'k9l1.pk', 'wb') as file:
    pk.dump({'x_trn': k9l1, 'x_tst': k9l1_tst, 'lab_tst': lab2}, file)

with open(pth + 'l9b1.pk', 'wb') as file:
    pk.dump({'x_trn': l9b1, 'x_tst': l9b1_tst, 'lab_tst': lab3}, file)



# need same entity for both train and test