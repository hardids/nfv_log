import numpy as np
import prc

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras

# 3. Tokenizer

def tokenizer(log):
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