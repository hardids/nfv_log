# import pandas as pd
import numpy as np
import prc

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import build_model
import explore_data
import vectorize_data
embedding_vector_length = 32


# # 1.DB or csv에서 알람기록 데이터 가져오기
db = prc.Makedata("./save_model/ipmi.csv")
db.labeling()
# (train_texts, train_labels), (val_texts, val_labels)
X_train, y_train = db.makeTrainset_feature()

# # 2. Train/Test Dataset 만들기

clean_logs_trainX = []
# clean_logs_testX = []

for log in X_train[:,0]:
    clean_logs_trainX.append(prc.preprocessing(log, remove_stopwords = True))
# for log in X_test[:,0]:
#     clean_logs_testX.append(prc.preprocessing(log, remove_stopwords = True))

# 3. Tokenizer
tokenizer = Tokenizer(220, oov_token='OOV')
tokenizer.fit_on_texts(clean_logs_trainX)

text_sequences_train = tokenizer.texts_to_sequences(clean_logs_trainX)
text_sequences_test = tokenizer.texts_to_sequences(clean_logs_testX)
word_vocab = tokenizer.word_index

data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) + 1
vocab_size = len(word_vocab) + 1


train_inputs = pad_sequences(text_sequences_train, maxlen=100, padding='post')
test_inputs = pad_sequences(text_sequences_test, maxlen=100, padding='post')
print('Shape of train data: ', train_inputs.shape)

X_train = np.concatenate([train_inputs, X_train[:,1:]], axis=1)
X_test = np.concatenate([test_inputs, X_test[:,1:]], axis=1)
X_train = X_train.astype(np.int64)
X_test = X_test.astype(np.int64)

# 4. Make Model

K.clear_session()
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=X_train.shape[1]) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
# model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss=sparse_categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=1e-3),metrics=['accuracy'])
model.summary()


# K.clear_session()
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_vector_length, input_length=X_train.shape[1]))
# # model.add(SpatialDropout1D(0.25))
# model.add(LSTM(128))
# # model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
# model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(lr=1e-3),metrics=['accuracy'])
# model.summary()


# 5. Model fit
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=8, callbacks=[es,mc], class_weight={0:1, 1:2, 2:5})