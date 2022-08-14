#%%
import pandas as pd
import numpy as np
import kerastuner as kt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as confmat
from sklearn.metrics import recall_score
from tensorflow import keras

def CleanData(data):
    for i in range(len(data.cluster)):
        if(data.cluster[i] == '´Day'): 
            data.cluster[i] = 'Day'
        elif(data.cluster[i] == 'Slot '):
            data.cluster[i] = 'Slot'
        elif(data.cluster[i] == 'projeto'):
            data.cluster[i] = 'Projeto'
        elif(data.cluster[i] == np.nan):
            data.cluster[i] = '-'
    
def BuildClassifierModel(hp):
    #Import pre-trained models
    # bert_preprocess=hub.KerasLayer("https://tfhub.dev/jeongukjae/distilbert_multi_cased_preprocess/2")
    # bert_encoder=hub.KerasLayer("https://tfhub.dev/jeongukjae/distilbert_multi_cased_L-6_H-768_A-12/1")
    bert_preprocess=hub.KerasLayer("https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_preprocess/1")
    bert_encoder=hub.KerasLayer("https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_L-24_H-1024_A-16/1")

    #Bert Layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    #Neural network layers
    hp_dropout = hp.Float("dropout", min_value=0, max_value=0.3, step=0.1)
    l = tf.keras.layers.Dropout(hp_dropout)(outputs['pooled_output'])
    for i in range(1, hp.Int("num_layers", 2, 5)):
        l = tf.keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=128, max_value=1024, step=128),
                activation="relu")(l)
        l = tf.keras.layers.Dropout(hp.Float("dropout_" + str(i), min_value=0, max_value=0.3, step=0.1))(l)
    l = tf.keras.layers.Dense(11, activation="softmax")(l)

    #Construct final model
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = tf.keras.Model(inputs=[text_input], outputs=[l])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return model

#%% Preparar os dados
train_data = pd.read_csv(r'Insert path to Data.csv here')

CleanData(train_data)
cluster_to_int = pd.DataFrame({'cluster': train_data.cluster.unique(), 'value': range(len(train_data.cluster.unique()))})

for i in range(len(cluster_to_int)):
    train_data.cluster = train_data.cluster.replace(to_replace = cluster_to_int.cluster[i], value = cluster_to_int.value[i])
del i

train_data.sample(frac=1).reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(train_data.melhorar, train_data.cluster, stratify=train_data.cluster)
x_train = x_train.to_numpy(dtype='str')
y_train = y_train.to_numpy(dtype='int16')
x_test = x_test.to_numpy(dtype='str')
y_test = y_test.to_numpy(dtype='int16')

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
del x_train, x_test, y_train, y_test, train_data

#%% Otimizar o modelo
tuner = kt.Hyperband(BuildClassifierModel,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x, y, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

#%%Encontre o número ótimo de épocas
model = tuner.hypermodel.build(best_hps)
history = model.fit(x, y, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

#%% Reinstanciar o hipermodelo e treiná-lo com o número ideal de épocas
hypermodel = tuner.hypermodel.build(best_hps)

hypermodel.fit(x, y, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)