from .nbr_base import NBRBase
from utils.metrics import *
import os
seed_value = 12321
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.layers import Input,multiply ,Dense, Dropout, Embedding,Concatenate, Reshape,Flatten,LSTM, Attention, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score


class MLPv12(NBRBase):
    def __init__(self, train_baskets, test_baskets,valid_baskets,dataset ,basket_count_min=0, min_item_count = 5, user_embed_size = 32,item_embed_size = 128,h1 = 128,h2 = 128,h3 = 128,h4 = 128,h5 = 128,history_len = 40, job_id = 1):
        super().__init__(train_baskets,test_baskets,valid_baskets,basket_count_min)
        self.model_name = dataset+ 'simple_mlpv12'
        self.dataset = dataset
        self.all_items = self.train_baskets[['item_id']].drop_duplicates()['item_id'].tolist()
        self.all_users = self.train_baskets[['user_id']].drop_duplicates()['user_id'].tolist()
        self.num_items = len(self.all_items) +1
        self.num_users = len(self.all_users) +1

        print("items:", self.num_items)
        item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name = 'item_count').reset_index()
        item_counts = item_counts[item_counts['item_count']>= min_item_count]
        item_counts_dict = dict(zip(item_counts['item_id'],item_counts['item_count']))
        print("filtered items:", len(item_counts_dict))
        self.num_items = len(item_counts_dict) +1
        self.item_id_mapper = {}
        self.id_item_mapper = {}
        self.user_id_mapper = {}
        self.id_user_mapper = {}

        counter = 0
        for i in range(len(self.all_items)):
            if self.all_items[i] in item_counts_dict:
                self.item_id_mapper[self.all_items[i]] = counter+1
                self.id_item_mapper[counter+1] = self.all_items[i]
                counter+=1
        for i in range(len(self.all_users)):
            self.user_id_mapper[self.all_users[i]] = i+1
            self.id_user_mapper[i+1] = self.all_users[i]

        self.user_embed_size = user_embed_size#32
        self.item_embed_size = item_embed_size#128
        #self.hidden_size = hidden_size#128
        self.history_len = history_len#40
        self.num_layers = 3

        self.data_path = self.model_name+'_'+str(job_id) + '_' + str(self.user_embed_size) + '_' + \
                         str(self.item_embed_size) + '_'+ str(h1) + '_'+ str(h2) + '_'+ str(h3) + '_'+ str(h4) + '_'+ str(h5) + '_' + str(self.history_len) + '_' + \
                         str(self.num_layers)

        input1 = Input(shape=(1,))
        input2 = Input(shape=(1,))
        input3 = Input(shape=(self.history_len,))
        input4 = Input(shape=(self.history_len,))

        x1 = Embedding(self.num_items, self.item_embed_size , input_length=1)(input1)
        x2 = Embedding(self.num_users, self.user_embed_size, input_length=1)(input2)
        
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x3 = input3
        x4 = input4

        x11 = Dense(h1, activation= 'relu')(Concatenate()([x1,x2]))
        x12 = tf.keras.layers.RepeatVector(self.history_len)(x11)
        x14 = Reshape((self.history_len,1))(x4)
        x14 = Dense(h1, activation= 'relu')(Concatenate()([x12,x14]))

        x = LSTM(h2,return_sequences = True)(x14, mask = tf.dtypes.cast(input4, tf.bool))
        x = LSTM(h3)(x, mask = tf.dtypes.cast(input4, tf.bool))


        x = Dense(h4, activation='relu')(x)
        x = Dense(h5, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        self.model = Model([input1,input2,input3,input4], output)

    def create_train_data(self):
        print(self.data_path)
        if os.path.isfile(self.model_name +'_' + str(self.history_len) + '_train_users.npy'):
            train_users = np.load(self.model_name +'_' + str(self.history_len) + '_train_users.npy')
            train_items = np.load(self.model_name +'_' + str(self.history_len) + '_train_items.npy')
            train_history = np.load(self.model_name +'_' + str(self.history_len) + '_train_history.npy')
            train_history2 = np.load(self.model_name +'_' + str(self.history_len) + '_train_history2.npy')
            train_labels = np.load(self.model_name +'_' + str(self.history_len)+ '_train_labels.npy')
            return train_items,train_users, train_history,train_history2, train_labels

        basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['item_id']))
        basket_items_dict['null'] = []

        user_baskets = self.train_baskets[['user_id','date','basket_id']].drop_duplicates().\
            sort_values(['user_id','date'],ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()

        user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['basket_id']))


        train_users = []
        train_items = []
        train_history = []
        train_history2 = []
        train_labels = []
        print('num users:', len(self.test_users))

        for c,user in enumerate(self.test_users):
            if c % 1000 ==1:
                print(c , 'user passed')

            baskets = user_baskets_dict[user]
            item_seq = {}
            for i, basket in enumerate(baskets):
                for item in basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)


            for i in range(max(0,len(baskets)-50), len(baskets)):
                label_basket = baskets[i]
                all_history_baskets = baskets[:i]
                items = []
                for basket in all_history_baskets:
                    for item in basket_items_dict[basket]:
                        items.append(item)
                items = list(set(items))
                for item in items:
                    if item not in self.item_id_mapper:
                        continue
                    index = np.argmax(np.array(item_seq[item])>=i)
                    if np.max(np.array(item_seq[item])) < i:
                        index = len(item_seq[item])
                    input_history = item_seq[item][:index].copy()
                    if len(input_history) ==0:
                        continue
                    if len(input_history) ==1 and input_history[0]==-1:
                        continue
                    while len(input_history) < self.history_len:
                        input_history.insert(0,-1)
                    real_input_history = []
                    for x in input_history:
                        if x == -1:
                            real_input_history.append(0)
                        else:
                            real_input_history.append(i-x)
                    real_input_history2 = []
                    for j,x in enumerate(input_history[:-1]):
                        if x == -1:
                            real_input_history2.append(0)
                        else:
                            real_input_history2.append(input_history[j+1]-input_history[j])
                    real_input_history2.append(i-input_history[-1])
                    train_users.append(self.user_id_mapper[user])
                    train_items.append(self.item_id_mapper[item])
                    train_history.append(real_input_history[-self.history_len:])
                    train_history2.append(real_input_history2[-self.history_len:])
                    #print(item, basket_items_dict[label_basket])
                    train_labels.append(float(item in basket_items_dict[label_basket]))

        train_items = np.array(train_items)
        train_users = np.array(train_users)
        train_history = np.array(train_history)
        train_history2 = np.array(train_history2)
        train_labels = np.array(train_labels)
        random_indices = np.random.choice(range(len(train_items)), len(train_items),replace=False)
        train_items = train_items[random_indices]
        train_users = train_users[random_indices]
        train_history = train_history[random_indices]
        train_history2 = train_history2[random_indices]
        train_labels = train_labels[random_indices]        

        np.save(self.model_name +'_' + str(self.history_len) + '_train_items.npy',train_items)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_users.npy',train_users)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_history.npy',train_history)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_history2.npy',train_history2)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_labels.npy',train_labels)

        return train_items,train_users, train_history,train_history2 ,train_labels

    def train(self):
        train_items, train_users, train_history,train_history2, train_labels = self.create_train_data()
        print(train_history.shape)
        print(np.count_nonzero(train_labels))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.data_path+'_weights.{epoch:02d}.hdf5',
            save_weights_only=True,
            save_best_only=False)

        self.model.compile(loss='binary_crossentropy',#'mean_squared_error',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        print(self.model.summary())
        history = self.model.fit([train_items,train_users,train_history,train_history2],train_labels, validation_split = None,
                                 batch_size=10000, epochs=5,shuffle=True, callbacks=[model_checkpoint_callback])#, class_weight= {0:1, 1:100})
        print("Training completed")

    def create_test_data(self,test_data='test'):
        if os.path.isfile(self.model_name +'_' + str(self.history_len)+ '_'+test_data+'_users.npy'):
            test_users = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_users.npy')
            test_items = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_items.npy')
            test_history = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history.npy')
            test_history2 = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history2.npy')
            test_labels = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_labels.npy')
            return test_items,test_users, test_history,test_history2, test_labels
        train_basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        train_basket_items_dict = dict(zip(train_basket_items['basket_id'],train_basket_items['item_id']))

        train_user_baskets = self.train_baskets[['user_id','date','basket_id']].drop_duplicates(). \
            sort_values(['user_id','date'],ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()
        train_user_baskets_dict = dict(zip(train_user_baskets['user_id'],train_user_baskets['basket_id']))

        train_user_items = self.train_baskets[['user_id','item_id']].drop_duplicates().groupby(['user_id'])['item_id'] \
            .apply(list).reset_index()
        train_user_items_dict = dict(zip(train_user_items['user_id'],train_user_items['item_id']))

        test_user_items = None
        if test_data == 'test':
            test_user_items = self.test_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        else:
            test_user_items = self.valid_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        test_user_items_dict = dict(zip(test_user_items['user_id'],test_user_items['item_id']))

        test_users = []
        test_items = []
        test_history = []
        test_history2 = []
        test_labels = []

        train_basket_items_dict['null'] = []
        for c,user in enumerate(test_user_items_dict):
            if user not in train_user_baskets_dict:
                continue
            if c % 100 ==1:
                print(c , 'user passed')
                #break

            baskets = train_user_baskets_dict[user]
            item_seq = {}
            for i, basket in enumerate(baskets):
                for item in train_basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)


            label_items = test_user_items_dict[user]

            items = list(set(train_user_items_dict[user]))

            #print(len(history_baskets))
            for item in items:#train_user_items_dict[user]:
                if item not in self.item_id_mapper:
                    continue
                input_history = item_seq[item][-self.history_len:]
                if len(input_history) ==0:
                    continue
                if len(input_history) ==1 and input_history[0]==-1:
                    continue
                while len(input_history) < self.history_len:
                    input_history.insert(0,-1)
                real_input_history = []
                for x in input_history:
                    if x == -1:
                        real_input_history.append(0)
                    else:
                        real_input_history.append(len(baskets)-x)

                real_input_history2 = []
                for j,x in enumerate(input_history[:-1]):
                    if x == -1:
                        real_input_history2.append(0)
                    else:
                        real_input_history2.append(input_history[j+1]-input_history[j])
                real_input_history2.append(len(baskets)-input_history[-1])
                test_users.append(self.user_id_mapper[user])
                test_items.append(self.item_id_mapper[item])
                test_history.append(real_input_history)
                test_history2.append(real_input_history2)
                test_labels.append(float(item in label_items))

        test_items = np.array(test_items)
        test_users = np.array(test_users)
        test_history = np.array(test_history)
        test_history2 = np.array(test_history2)
        test_labels = np.array(test_labels)

        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_items.npy',test_items)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_users.npy',test_users)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history.npy',test_history)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history2.npy',test_history2)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_labels.npy',test_labels)

        return test_items,test_users, test_history,test_history2, test_labels

    def predict(self,epoch = '01'):
        test_items, test_users, test_history,test_history2, test_labels = self.create_test_data('test')
        valid_items, valid_users, valid_history,valid_history2 ,valid_labels = self.create_test_data('valid')
        user_valid_baskets_df = self.valid_baskets.groupby('user_id')['item_id'].apply(list).reset_index()
        user_valid_baskets_dict = dict(zip( user_valid_baskets_df['user_id'],user_valid_baskets_df['item_id']))

        epoch_recall = []
        for epoch in range(1,6):
            print('epoch',epoch)
            epoch_str = str(epoch)
            if epoch<10:
                epoch_str = '0'+str(epoch)

            self.model.load_weights(self.data_path+'_weights.'+epoch_str+'.hdf5')
            y_pred = self.model.predict([valid_items,valid_users,valid_history,valid_history2],batch_size = 5000)
            predictions = [round(value) for value in y_pred.flatten().tolist()]
            accuracy = accuracy_score(valid_labels, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            recall_scores = []
            for user in user_valid_baskets_dict:
                top_items = []
                if user in self.user_id_mapper:
                    user_id = self.user_id_mapper[user]
                    indices = np.argwhere(valid_users == user_id)
                    item_scores = y_pred[indices].flatten()
                    item_ids = valid_items[indices].flatten()

                    item_score_dic = {}
                    for i, item_id in enumerate(item_ids):
                        item_score_dic[self.id_item_mapper[item_id]] = item_scores[i]
                    sorted_item_scores = sorted(item_score_dic.items(), key= lambda x: x[1], reverse = True)
                    top_items = [x[0] for x in sorted_item_scores]
                recall_scores.append(recall_k(user_valid_baskets_dict[user],top_items,
                                              len(user_valid_baskets_dict[user])))
            epoch_recall.append(np.mean(recall_scores))
        print(epoch_recall)
        print(np.argmax(np.array(epoch_recall)))
        best_epoch = np.argmax(np.array(epoch_recall)) + 1
        epoch_str = str(best_epoch)
        if best_epoch<10:
            epoch_str = '0'+str(best_epoch)
        print('best model:',self.data_path+'_weights.'+epoch_str+'.hdf5')
        print('best recall on valid:',epoch_recall[best_epoch-1])
        self.model.load_weights(self.data_path+'_weights.'+epoch_str+'.hdf5')
        y_pred = self.model.predict([test_items,test_users,test_history,test_history2],batch_size = 5000)
        prediction_baskets = {}
        prediction_scores = {}
        for user in self.test_users:
            top_items = []
            if user in self.user_id_mapper:
                user_id = self.user_id_mapper[user]
                indices = np.argwhere(test_users == user_id)
                item_scores = y_pred[indices].flatten()
                item_ids = test_items[indices].flatten()
                item_score_dic = {}
                for i, item_id in enumerate(item_ids):
                    item_score_dic[self.id_item_mapper[item_id]] = item_scores[i]
                sorted_item_scores = sorted(item_score_dic.items(), key= lambda x: x[1], reverse = True)
                top_items = [x[0] for x in sorted_item_scores]
                prediction_scores[user] = sorted_item_scores
            
            prediction_baskets[user] = top_items

        return prediction_baskets
