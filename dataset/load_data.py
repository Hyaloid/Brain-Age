# coding=utf8
import pandas as pd
import os


cur_dir = os.getcwd()
print(cur_dir)
os.chdir('./dataset')


train = pd.read_csv('input/data_train.csv')
test = pd.read_csv('input/data_test.csv')
