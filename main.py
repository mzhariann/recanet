import pandas as pd
import sys
from models.mlp_v12 import MLPv12
from utils.metrics import *
seed_value = 12321
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
from tensorflow.keras import backend as K
K.clear_session()
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='ReCANet')
    parser.add_argument('-dataset', type=str, default='dunnhumby_cj')
    parser.add_argument('-user_embed_size', type=int, default=32)
    parser.add_argument('-item_embed_size', type=int, default=128)
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-history_len', type=int, default=20)
    parser.add_argument('-job_id', type=int, default=10)
    args = parser.parse_args()
    return args


args = parse_args()
dataset = args.dataset

data_path = 'data/'
#dataset = 'tafeng'
#dataset = 'dunnhumby_cj'
#dataset = 'instacart_small_sample'
#dataset = 'valued_shopper_small_sample'

print(dataset)
train_baskets = pd.read_csv(data_path+dataset+'/train_baskets.csv')
test_baskets = pd.read_csv(data_path+dataset+'/test_baskets.csv')
valid_baskets = pd.read_csv(data_path+dataset+'/valid_baskets.csv')
print('data read')
user_test_baskets_df = test_baskets.groupby('user_id')['item_id'].apply(list).reset_index()
user_test_baskets_dict = dict(zip( user_test_baskets_df['user_id'],user_test_baskets_df['item_id']))
model = MLPv12(train_baskets, test_baskets,valid_baskets,data_path+dataset+'/' ,3,  5,  args.user_embed_size,args.item_embed_size,64,64,64,64,64,args.history_len, job_id = args.job_id)


model.train()
print('model trained')

user_predictions = model.predict()
final_users = set(model.test_users).intersection(set(list(user_test_baskets_dict.keys())))
print('predictions ready',len(user_predictions))
print('number of final test users:',len(final_users))
for k in [5,10,20,'B']:
    print(k)
    recall_scores = {}
    ndcg_scores = {}
    zero = 0
    for user in final_users:

        top_items = []
        if user in user_predictions:
            top_items = user_predictions[user]
        else:
            zero+=1

        if k == 'B':
            recall_scores[user] = recall_k(user_test_baskets_dict[user],top_items,len(user_test_baskets_dict[user]))
            ndcg_scores[user] = ndcg_k(user_test_baskets_dict[user],top_items,len(user_test_baskets_dict[user]))
        else:
            recall_scores[user] = recall_k(user_test_baskets_dict[user],top_items,k)
            ndcg_scores[user] = ndcg_k(user_test_baskets_dict[user],top_items,k)
    #print(zero)
    print('recall:',np.mean(list(recall_scores.values())))
    print('ndcg:',np.mean(list(ndcg_scores.values())))

