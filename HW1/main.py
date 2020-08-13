#%%
import pandas as pd
import matplotlib.pyplot as plt
import csv
train_file_name = "train.csv"
test_file_name = "test.csv"
train_days = 9
features = 18
label_pm25_idx = 9
test_feature = 0

def get_file_data(file_name):
    row_data = []
    with open(file_name, encoding="big5") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        title = True
        for row in csv_reader:
            if title:
                title = False
                continue
            row_data.append(row)
    return row_data

row_data = get_file_data(train_file_name)
# %%
def clean_data(row_data):
    train_data = [[] for i in range(features)] 
    idx = 0
    for row in row_data:
        for d in row[3:]:
            if d == 'NR':
                d = 0
            train_data[idx].append(float(d))
        idx = idx + 1
        idx = idx % features
    return train_data
train_data = clean_data(row_data)
# %%
pm2_5 = []
def get_label(train_data):
    pm2_5 = []
    for data in train_data[label_pm25_idx]:
        pm2_5.append(float(data))
    return pm2_5
pm2_5 = get_label(train_data)
# %%
import numpy as np
import random
from statistics import mean
from statistics import stdev

learning_rate = 0.2

w = random.sample(range(1, 10), train_days)
w = [i/10 for i in w]
b = 0.0
grad_w = 0.0
grad_b = 0.0
data_num = len(train_data[test_feature]) - train_days - 1
for idx in range(data_num):   
    # feature scaling
    x = train_data[test_feature][idx:idx+train_days]
    m = mean(x)
    std = stdev(x) + 0.01
    x = [(i-m)/std for i in x]
    
    # GD training
    y_hat = 0.0
    for i in range(train_days):
        y_hat += (w[i] * x[i])
    y_hat += b
    for i in range(train_days):
        grad_w = -2/5750 * (pm2_5[idx] - y_hat) * x[i]
        w[i] = w[i] - learning_rate*grad_w
    grad_b = -2/5750 * (pm2_5[idx] - y_hat)
    b = b - learning_rate*grad_b

# %%
file_name = "test.csv"
row_data = []
with open(file_name, encoding="big5") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_data.append(row)
y_hat = 0
test_x = row_data[test_feature][2:]
test_x = [float(i) for i in test_x]


m = mean(test_x)
std = stdev(test_x) + 0.01
x = [(i-m)/std for i in test_x]
# x = test_x
# print(x)
for i in range(train_days):
    y_hat += (w[i] * x[i])
y_hat += b

# %%
# print((y_hat*std)+m)
print(y_hat)
# %%
