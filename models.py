import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import tensorflow_addons as tfa 

from keras import Model, layers
from keras.layers import Dense, concatenate, Input,  Dropout
from keras import regularizers, initializers


def MLPmodel(feat_n,modelnum): 
    input = layers.Input(shape=(152, feat_n))
    layer1 = layers.Flatten()(input)
    layer2 = Dropout(rate=0.1)(layer1)
    layer3 = layers.Dense(8, kernel_regularizer='l1_l2', activation='relu', kernel_initializer='he_normal')(layer2)
    layer4 = layers.Dense(8, kernel_regularizer='l1_l2', activation='relu', kernel_initializer='he_normal')(layer3)
    output = layers.Dense(4, activation='softmax', kernel_initializer='he_normal')(layer3)

    model = Model(input, output, name='model2_{}'.format(modelnum))

    return model

def CNNmodel(feat_n,modelnum): 
    input = layers.Input(shape=(152, feat_n))
    layer1 = layers.Conv1D(32, kernel_size=(3), activation='relu', input_shape=(152,feat_n), kernel_initializer='he_normal')(input)
    layer2 = layers.Conv1D(16, (3), activation='relu', kernel_regularizer='l1_l2', kernel_initializer='he_normal')(layer1)
    layer3 = layers.MaxPooling1D(pool_size=(2))(layer2)
    layer4 = Dropout(rate=0.1)(layer3)
    layer5 = layers.Flatten()(layer4)
    layer6 = Dense(4, activation='relu', kernel_regularizer='l1_l2', kernel_initializer='he_normal')(layer5)
    layer7 = Dense(16, activation='relu', kernel_regularizer='l1_l2', kernel_initializer='he_normal')(layer6)
    output = layers.Dense(4, activation='softmax', kernel_initializer='he_normal')(layer7)

    model = Model(input, output, name='CNN2_{}'.format(modelnum))

    return model

def Ensemble(seq_n, feat_n, modelnum): 
    input1 = layers.Input(shape=(4000, seq_n))
    layer1 = layers.LSTM(units=32,
                    input_shape=(4000, seq_n),
                    return_sequences=True,
            kernel_regularizer='l1_l2', kernel_initializer='he_normal')(input1)#regularizers.L2(0.01)
    layer2 = layers.LSTM(units=8,
                    return_sequences=False,
            kernel_regularizer='l1_l2', kernel_initializer='he_normal')(layer1)
    dropout = Dropout(rate=0.3)(layer2)
    output1 = Dense(8, kernel_initializer='he_normal',activation='relu')(dropout)

    input2 = layers.Input(shape=(152, feat_n))
    layer3 = layers.Conv1D(32, kernel_size=(3), activation='relu', input_shape=(152,feat_n), 
            kernel_regularizer='l1_l2',kernel_initializer='he_normal')(input2)
    layer4 = layers.Conv1D(16, (3), activation='relu', 
            kernel_regularizer='l1_l2',kernel_initializer='he_normal')(layer3)
    layer5 = layers.MaxPooling1D(pool_size=(2))(layer4)
    layer6 = Dropout(rate=0.1)(layer5)
    layer7 = layers.Flatten()(layer6)
    layer8 = Dense(4, activation='relu', 
            kernel_regularizer='l1_l2',kernel_initializer='he_normal')(layer7)
    layer9 = Dense(16, activation='relu', 
            kernel_regularizer='l1_l2',kernel_initializer='he_normal')(layer8)
    output2 = layers.Dense(8, activation='relu',kernel_initializer='he_normal')(layer9)

    concat_model = concatenate([output1, output2], axis=-1) 
    concat_model = layers.Dense(16, activation='relu', 
            kernel_regularizer='l1_l2',kernel_initializer='he_normal')(concat_model)
    concat_out = layers.Dense(4, activation='softmax')(concat_model)

    model = Model([input1, input2], concat_out, name='model3_{}'.format(modelnum))

    return model

def seqModel(seq_n, modelnum): 
    input = layers.Input(shape=(4000, seq_n))
    layer1 = layers.LSTM(units=32,
                    input_shape=(4000, seq_n), kernel_initializer='he_normal',
                    kernel_regularizer='l1_l2',return_sequences=True)(input)
    layer2 = layers.LSTM(units=8,                    
                    kernel_regularizer='l1_l2',return_sequences=True, kernel_initializer='he_normal')(layer1)
    layer3 = layers.LSTM(units=8,
                    kernel_regularizer='l1_l2',return_sequences=False, kernel_initializer='he_normal')(layer2)
    dropout = tf.keras.layers.Dropout(rate=0.3)(layer3)
    layer4 = Dense(16, activation='relu', kernel_initializer='he_normal')(dropout)
    output = layers.Dense(4, activation='softmax')(layer4)

    model = Model(inputs=input, outputs=output, name='model{}'.format(modelnum))


    return model

def sl_feature(modelnum):
    input1 = layers.Input(shape=(41, 24))
    layer1 = layers.Conv1D(32, kernel_size=(3), input_shape=(41, 24), activation='relu', 
            kernel_regularizer='l1_l2',kernel_initializer='he_normal')(input1)
    layer2 = layers.AveragePooling1D(pool_size = (2), padding = 'valid')(layer1)
    layer3 = layers.Flatten()(layer2)
    layer4 = Dropout(rate=0.3)(layer3)
    output1 = layers.Dense(4, activation='relu', kernel_initializer='he_normal')(layer4)

    input2 = layers.Input(shape=(152, 2))
    layer5 = layers.Conv1D(32, kernel_size=(2), input_shape=(152,2), activation='relu', 
                kernel_regularizer='l1_l2', kernel_initializer='he_normal')(input2)
    layer6 = layers.AveragePooling1D(pool_size = (2), padding = 'valid')(layer5)
    layer7 = layers.Flatten()(layer6)
    layer8 = Dropout(rate=0.3)(layer7)
    output2 = layers.Dense(4, activation='relu', kernel_initializer='he_normal')(layer8)
    
    concat_input = concatenate([output1, output2], axis=-1)
    concat_model = layers.Dense(16, activation='relu', 
                kernel_regularizer='l1_l2', kernel_initializer='he_normal')(concat_input)
    concat_out = layers.Dense(4, activation='softmax')(concat_model)
    
    model = Model([input1, input2], concat_out, name='model{}'.format(modelnum))

    return model

def label_ensemble(seq_n):

    input1 = layers.Input(shape=(4000, seq_n))
    layer1 = layers.LSTM(units=32,
                    input_shape=(4000, seq_n), kernel_initializer='he_normal',
                    kernel_regularizer='l1_l2',return_sequences=True)(input1)
    layer2 = layers.LSTM(units=8, kernel_regularizer='l1_l2',
                    return_sequences=True, kernel_initializer='he_normal')(layer1)
    layer3 = layers.LSTM(units=8, kernel_regularizer='l1_l2',
                    return_sequences=False, kernel_initializer='he_normal')(layer2)
    dropout = tf.keras.layers.Dropout(rate=0.3)(layer3)
    layer4 = Dense(16, activation='relu', kernel_initializer='he_normal')(dropout)
    output1 = layers.Dense(4, activation='softmax')(layer4)

    input2 = layers.Input(shape=(41, 24))
    layer1 = layers.Conv1D(32, kernel_size=(3), input_shape=(41, 24), kernel_regularizer='l1_l2',
                    activation='relu', kernel_initializer='he_normal')(input2)
    layer2 = layers.Flatten()(layer1)
    layer3 = layers.Dense(16, kernel_regularizer='l1_l2',
                    activation='relu', kernel_initializer='he_normal')(layer2)
    layer4 = Dropout(rate=0.3)(layer3)
    layer5 = layers.Dense(16, kernel_regularizer='l1_l2',
                    activation='relu', kernel_initializer='he_normal')(layer4)
    output2 = layers.Dense(4, activation='relu', kernel_initializer='he_normal')(layer5)

    concat_input = concatenate([output1, output2], axis=-1)
    concat_model = layers.Dense(16, activation='relu', 
                kernel_regularizer='l1_l2', kernel_initializer='he_normal')(concat_input)
    concat_out = layers.Dense(4, activation='softmax')(concat_model)
    
    model = Model([input1, input2], concat_out, name='model4-3')

    return model

def labelfeature():
    print("****")
    input = layers.Input(shape=(41, 24))
    layer1 = layers.Conv1D(32, kernel_size=(3), input_shape=(41, 24),
             kernel_regularizer='l1_l2',activation='relu', kernel_initializer='he_normal')(input)
    layer2 = layers.Flatten()(layer1)
    layer3 = layers.Dense(16, kernel_regularizer='l1_l2',activation='relu', kernel_initializer='he_normal')(layer2)
    layer4 = Dropout(rate=0.3)(layer3)
    layer5 = layers.Dense(64, kernel_regularizer='l1_l2',activation='relu', kernel_initializer='he_normal')(layer4)

    layer6 = layers.Dense(16, kernel_regularizer='l1_l2',activation='relu', kernel_initializer='he_normal')(layer5)
    output = layers.Dense(4, activation='softmax', kernel_initializer='he_normal')(layer6)
    model = Model(input, output, name='model4-2')
    return model
