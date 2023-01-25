## PACKAGES
#!/usr/bin/env python3
import os

import pandas as pd
import numpy as np
from time import time
from keras import backend as K

import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import tensorflow_addons as tfa 
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from keras_preprocessing.sequence import pad_sequences
from keras import backend as K

from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
## Make X, Y

def makeXY():
    #folder = '/home/jiyoon/OneDrive/Projects/Thesis/assessmentdata/assessmentdata/'
    df = pd.read_pickle('./data.pkl')#.format(folder))

    df.columns
    data = df[['subactivity', 'sv', 'timegap', 'sensorvalues']]
    sv_dataset = data.values

    df['target'] = df.apply(lambda x : 1 if x['diagnosis']==1 else (2 if (x['diagnosis']==2) else 3), axis = 1)

    grp = df[['target']].values.reshape(1,-1)[0].tolist()
    data.info()
    documents = []
    for user in sv_dataset[:,-1]:
        user = " ".join(user)
        documents.append(user)
        
    cvectorizer = CountVectorizer()
    cdtm = cvectorizer.fit_transform(documents)
    tf_data = pd.DataFrame(cdtm.toarray(), columns = cvectorizer.get_feature_names_out())

    ivectorizer = TfidfVectorizer()
    idtm = ivectorizer.fit_transform(documents)
    tfidf_data = pd.DataFrame(idtm.toarray(), columns = ivectorizer.get_feature_names_out())
    X_f=[]
    for user in range(332):
        get_feature = np.array([tf_data.iloc[user], tfidf_data.iloc[user]])
        get_feature = get_feature.transpose()
        X_f.append(get_feature)
    X_f = np.array(X_f)
    print('X_f.shape', X_f.shape)
    X_seq = []

    for user in sv_dataset[:,:3]:
        seq = []
        for i, f in enumerate(user):
            f=np.array(f)
            f=f.reshape(1, f.shape[0])
            paded_seq = pad_sequences(f, maxlen=4000, dtype='float64', value=0.0, padding='post', truncating='pre')
            paded_seq = paded_seq.reshape(4000, paded_seq.shape[0])
            seq.append(paded_seq)
        seq = np.array(seq)
        seq = seq.squeeze().transpose()
        X_seq.append(seq)
    X_seq = np.array(X_seq)
    print('X_seq.shape', X_seq.shape)
    target=df['target'].values #20 56 256

    target = np.array(target)
    target = target.reshape(-1, 1)
    target = to_categorical(target)
    print('target.shape: ',target.shape)

    return X_seq, X_f, target, grp



def make_lf(X_seq):
    subactivity = X_seq[:, :, 0]
    subactivity = subactivity.reshape(subactivity.shape[0], subactivity.shape[1], 1)
    subactivity = np.array([np.array(seq) for seq in subactivity])
    print('subactivity.shape: ', subactivity.shape)
    top_array = [[[0 for col in range(24)] for row in range(41)] for batch in range(332)] #array[start:0, end:1, sub_check:2~14, sub_seq:15~40][task]
    sub_sequence = { 't1' : [0], 't2': [0], 't3': [0], 't4': [0], 't5': [0],
    't6': [0], 't7': [0], 't8': [0], 't9': [0], 't10': [0], 't11': [0], 
    't12': [0], 't13': [0], 't14': [0], 't15': [0], 't16': [0], 't17': [0], 
    't18': [0], 't19': [0], 't20': [0], 't21': [0], 't22': [0], 't22': [0], 't23': [0], 't24': [0]}
    #print('top_array.shape: ',np.array(top_array).shape)
    bottom_array = np.empty((332, 26, 24), int) 
    for id, patient in enumerate(subactivity):
        sub_sequence_array = np.empty((26,0), int)#[[0 for col in range(24)] for row in range(26)]
        for row in patient:
            l = str(row[0])
            task = int(l.split('.')[0])

            #subactivity start:0.001, incomplete:0.004, end:0.009
            if len(l)==5:
                code = l[-1]
                if code == 1: #start
                    top_array[id][0][task]=1
                elif code == 9: #end
                    top_array[id][1][task]=1
                elif code == 4: #incomplete
                    top_array[id][1][task]=0
                elif int(row[0])==0:
                    pass
            else: #ex: 1.10, 16.7, 0.00
                if int(row[0])==0:
                    pass
                elif int(row[0]) > 16:
                    pass
                else:
                    sub_l = int(l.split('.')[1]) #ex.10
                    top_array[id][1+sub_l][task]+=1 
                    sub_sequence['t{}'.format(task)].append(sub_l)

        for k in sub_sequence.keys():
            value = sub_sequence[k]
            task=int(str(k)[1:])
            task_seq_array = pad_sequences(np.array([value]), maxlen=26, dtype='float64', value=0.0, padding='post')
            task_seq_array = task_seq_array.reshape(task_seq_array.shape[1], task_seq_array.shape[0])
            sub_sequence_array = np.append(sub_sequence_array, task_seq_array, axis=1)###
            if sub_sequence_array.shape == (26,24):
                bottom_array[id] = sub_sequence_array
                #print(bottom_array.shape, bottom_array[id].shape)
        top_array[id][15:][:] = bottom_array

    final_derivation = np.array(top_array)
    X_lf = final_derivation.reshape(332,41,24)
    return X_lf


def scale_data(train, test, len):
    sc = StandardScaler()
    for ss in range(len):
        sc.partial_fit(train[:, ss, :])

    results=[]
    for ss in range(len):
        results.append(sc.transform(train[:, ss, :]).reshape(train.shape[0], 1, train.shape[2]))
    scaled_train = np.concatenate(results, axis=1)
    #print('scaled_train.shape: ', scaled_train.shape)

    results=[]
    for ss in range(len):
        results.append(sc.transform(test[:, ss, :]).reshape(test.shape[0], 1, test.shape[2]))
    scaled_test = np.concatenate(results, axis=1)
    #print('scaled_test.shape: ', scaled_test.shape)

    return scaled_train, scaled_test

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

def show_his(history):
    fig,(ax0, ax1, ax2) = plt.subplots(nrows = 1, ncols = 3, sharey = False, figsize=(16,5), dpi=80)
    ax0.plot(history.history["loss"], c='red', label = 'loss')
    ax0.plot(history.history["val_loss"], c='blue', label = 'val_loss')
    ax0.legend(loc='upper right')
    ax0.set(title='loss')
    ax1.plot(history.history["accuracy"], c='red', label = 'accuracy')
    ax1.plot(history.history["val_accuracy"], c='blue', label = 'val_accuracy')
    ax1.legend(loc='upper right')
    ax1.set(title='accuracy')
    ax2.plot(history.history["f1_score"], c='red', label = 'f1')
    ax2.plot(history.history["val_f1_score"], c='blue', label = 'val_f1_score')
    ax2.legend(loc='upper right')    
    ax2.set(title='f1_score')
    plt.show()
    return plt

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed
    
f1 = tfa.metrics.F1Score(num_classes=4, 
                            average='weighted', 
                            dtype=tf.float32)

def callback(model, fold_var):
    #reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.8, patience=30, max_lr=0.0000001, verbose=1)
    csv_logger = CSVLogger('./{}_{}.csv'.format(model.name,fold_var))
    checkpoint = ModelCheckpoint('./{}_{}.h5'.format(model.name, fold_var),
                                     save_best_only=True, monitor='loss')
    early_stopping = EarlyStopping(monitor='accuracy', patience=30, verbose=2, restore_best_weights=True)
    return [csv_logger, checkpoint, early_stopping]


def run_history(model, X, y, fold_var, epoch, bat):
    model.compile(optimizer=optimizers.Nadam(learning_rate= 1e-5), loss=focal_loss(), metrics=['accuracy', f1])
    model.summary()
    start = time()
    print("*** fold: ",fold_var)
    history = model.fit(X, y,
            epochs=epoch, batch_size=bat, validation_split=0.1, 
            callbacks=callback(model, fold_var), verbose=0)
    train_time = (time()-start)
    print(train_time)
    return model, history, train_time

def measures(val_loss, val_acc, val_f1, timecost):
    print('test_acc(mean, std, max, min):', np.mean(val_acc), np.std(val_acc), np.max(val_acc), np.min(val_acc))
    print('test_f1(mean, std, max, min):', np.mean(val_f1), np.std(val_f1), np.max(val_f1), np.min(val_f1))
    print('train_timecost(mean, std, max, min):', np.mean(timecost), np.std(timecost), np.max(timecost), np.min(timecost))
    print('train_loss(mean, std, max, min)(mean, std, max, min):', np.mean(val_loss), np.std(val_loss), np.max(val_loss), np.min(val_loss))
    print(val_loss, val_acc, val_f1, timecost)

    return val_loss, val_acc, val_f1, timecost