import pandas as pd
import numpy as np

'''
Reads the raw files, renames columns, last basket as test and the rest as train.
No additional preprocessing steps.
'''


prior_orders_file_path = 'instacart_2017_05_01/order_products__prior.csv'
train_orders_file_path = 'instacart_2017_05_01/order_products__train.csv'
orders_file_path = 'instacart_2017_05_01/orders.csv'
train_baskets_file_path = 'data/instacart/train_baskets.csv'
test_baskets_file_path = 'data/instacart/test_baskets.csv'
valid_baskets_file_path = 'data/instacart/valid_baskets.csv'

prior_orders = pd.read_csv(prior_orders_file_path)
train_orders = pd.read_csv(train_orders_file_path)
all_orders = pd.concat([prior_orders,train_orders])
#print(all_orders.shape)
#print(all_orders.nunique())

order_info = pd.read_csv(orders_file_path)

all_orders = pd.merge(order_info,all_orders,how='inner')
print(all_orders.shape)
print(all_orders.nunique())
#print(all_orders.head())

all_orders = all_orders.rename(columns={'order_id':'basket_id', 'product_id':'item_id'})



#all_users = list(set(all_orders['user_id'].tolist()))
#random_users_indices = np.random.choice(range(len(all_users)), 10000, replace=False)
#random_users = [all_users[i] for i in range(len(all_users)) if i in random_users_indices]
#all_orders = all_orders[all_orders['user_id'].isin(random_users)]


last_baskets = all_orders[['user_id','basket_id','order_number']].drop_duplicates() \
    .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'order_number'))
last_baskets.index = last_baskets.index.droplevel()
test_baskets = pd.merge(last_baskets, all_orders, how='left')
train_baskets = pd.concat([all_orders,test_baskets]).drop_duplicates(keep=False)

all_users = list(set(test_baskets['user_id'].tolist()))
valid_indices = np.random.choice(range(len(all_users)),int(0.5*len(all_users)),
                                 replace=False)
valid_users = [all_users[i] for i in valid_indices]

valid_baskets = test_baskets[test_baskets['user_id'].isin(valid_users)]
test_baskets = test_baskets[~test_baskets['user_id'].isin(valid_users)]

print(valid_baskets.shape)
print(test_baskets.shape)
print(train_baskets.shape)

print(valid_baskets.nunique())
print(test_baskets.nunique())
print(train_baskets.nunique())

train_baskets.to_csv(train_baskets_file_path,index=False)
test_baskets.to_csv(test_baskets_file_path,index=False)
valid_baskets.to_csv(valid_baskets_file_path,index=False)
