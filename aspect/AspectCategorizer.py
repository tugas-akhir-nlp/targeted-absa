from Preprocessor import Preprocessor
from Preprocessor import MAX_LENGTH, EMBEDDING_SIZE, ASPECTS

import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from keras import backend as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D

# https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code 
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + k.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + k.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+k.epsilon()))

class AspectCategorizer():
    config_file = 'config.json'
    weight_file = 'model.h5'

    def __init__ (
            self,
            normalize = False,
            lowercase = True,
            remove_punct = True,
            masking = False,
            embedding = True,
            trainable_embedding = True,
            pos_tag = None,
            dependency = None,
            use_rnn = True,
            rnn_type = 'lstm',
            use_cnn = False,
            use_svm = False,
            use_stacked_svm = False,
            use_attention = False,
            n_neuron = 128,
            n_dense = 1,
            dropout = 0.5,
            optimizer = 'adam'):
        self.preprocessor  =  Preprocessor(
            normalize = normalize,
            lowercase = lowercase,
            remove_punct = remove_punct,
            masking = masking
        )
        self.tokenizer = self.preprocessor.get_tokenized()
        self.aspects = ASPECTS
        self.model = None
        self.history = None

        self.embedding = embedding
        self.trainable_embedding = trainable_embedding
        self.pos_tag = pos_tag
        self.dependency = dependency
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.use_cnn = use_cnn
        self.use_svm = use_svm
        self.use_stacked_svm = use_stacked_svm
        self.use_attention = use_attention
        self.n_neuron = n_neuron 
        self.n_dense = n_dense 
        self.dropout = dropout
        self.optimizer = optimizer

        print("success")

    def __build_model(self):
        vocab_size = self.preprocessor.get_vocab_size(self.tokenizer)

        if self.embedding:
            embedding_matrix = self.preprocessor.get_embedding_matrix(self.tokenizer)
            main_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='main_input')
            x = Embedding(
                output_dim=EMBEDDING_SIZE,
                input_dim=vocab_size,
                input_length=MAX_LENGTH,
                weights=[embedding_matrix],
                trainable=self.trainable_embedding,
            )(main_input)

            if self.pos_tag is 'embedding':
                pos_matrix = self.preprocessor.get_pos_matrix(review)
                pos_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='pos_input')
                x2 = Embedding(
                    output_dim=30,
                    input_dim=pos_size,
                    input_length=MAX_LENGTH,
                    weights=[pos_matrix],
                    trainable=self.trainable_embedding,
                )(pos_input)
                x = keras.layers.concatenate([x, x2])

        else:
            embedded_input = self.preprocessor.get_embedded_input(review)

        if self.pos_tag is 'one_hot':
            encoded_pos = self.preprocessor.get_encoded_pos(review)

        if self.use_rnn is True:
            if self.rnn_type is 'gru':
                x = Bidirectional(GRU(self.n_neuron, return_sequences=True))(x)
            else:
                x = Bidirectional(LSTM(self.n_neuron, return_sequences=True))(x)
            x = GlobalMaxPool1D()(x)
            x = Dropout(self.dropout)(x)

        if self.use_cnn is True:
            pass

        if self.n_dense is not 0:
            for i in range(self.n_dense):
                x = Dense(self.n_neuron, activation='relu')(x)
                x = Dropout(self.dropout)(x)

        out = Dense(len(self.aspects), activation='sigmoid')(x)

        if self.pos_tag is 'embedding':
            model = Model([main_input, pos_input], out)
        else:
            model = Model(main_input, out)

        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=[f1]
        )

        return model

    def train(
        self,
        x_train,
        y_train,
        batch_size = 16,
        epochs = 5,
        verbose = 1,
        validation_split = 0.0,
        cross_validation = False,
        n_fold = 3,
        grid_search = False,
        callbacks = None):

        model = self.__build_model()

        if self.pos_tag is 'embedding':
            x_input = [x_train, pos_tag]
        else: 
            x_input = x_train

        history = model.fit(
            x = x_input, 
            y = y_train, 
            batch_size = batch_size,
            epochs = epochs, 
            verbose = verbose,
            validation_split = validation_split,
            callbacks = callbacks
        )

        self.model = model
        self.history = history

    def evaluate(self, x_train, y_train, x_test, y_test):
        x = [x_train, x_test, y_train, y_test]
        # x_pos = [pos_train, pos_test]
        x_name = ['Train-All', 'Test-All']

        print("======================= EVALUATION =======================")
        print('{:10s} {:10s} {:10s} {:10s}'.format('ASPECT', 'PREC', 'RECALL', 'F1'))

        for i in range(2):
            if self.pos_tag is 'embedding':
                y_pred = self.model.predict([x[i], x_pos[i]])
            else:
                y_pred = self.model.predict(x[i])
            y_true = x[i+2]

            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_true)

            y_pred = (y_pred>0.5).astype(int)
            y_true = (y_true>0).astype(int)

            acc = accuracy_score(y_true.reshape([-1]), y_pred.reshape([-1]))
            precision = precision_score(y_true, y_pred, average='micro')
            recall = recall_score(y_true, y_pred, average='micro')
            f1 = f1_score(y_true, y_pred, average='micro')

            print('{:10s} {:<10.4f} {:<10.4f} {:<10.4f} '.format(x_name[i], precision, recall, f1))

    def evaluate_each_aspect(self, x_test, y_test):
        if self.pos_tag is 'embedding':
            y_pred = self.model.predict([x_test, pos_test])
        else:
            y_pred = self.model.predict(x_test)
        y_true = y_test

        y_predlist = list()
        y_truelist = list()
        for i in range(len(self.aspects)):
            tmp = list()
            tmp = [item[i] for item in y_pred]
            y_predlist.append(tmp)
            tmp = list()
            tmp = [item[i] for item in y_true]
            y_truelist.append(tmp)
                
        for i in range(len(self.aspects)):
            y_predlist[i] = np.asarray(y_predlist[i])
            y_truelist[i] = np.asarray(y_truelist[i])

            y_predlist[i] = (y_predlist[i]>0.5).astype(int)
            y_truelist[i] = (y_truelist[i]>0).astype(int)

            acc = accuracy_score(y_truelist[i], y_predlist[i])
            precision = precision_score(y_truelist[i], y_predlist[i])
            recall = recall_score(y_truelist[i], y_predlist[i])
            f1 = f1_score(y_truelist[i], y_predlist[i])
            print('{:10s} {:<10.4f} {:<10.4f} {:<10.4f} '.format(self.aspects[i], precision, recall, f1))
        print('\n')


    def predict(self, new):
        encoded_new = self.tokenizer.texts_to_sequences([new.lower()])
        input_new = pad_sequences(encoded_new, maxlen=MAX_LENGTH, padding='post')
        pred = self.model.predict(input_new)
        
        label = list()
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                if (pred[i][j] > 0.5):
                    label.append(self.aspects[j])

        print("======================= PREDICTION =======================" )
        print(new)
        print(label)
        return label

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            print("Making the directory: {}".format(dir_path))
            os.mkdir(dir_path)
        self.model.save(os.path.join(dir_path, self.weight_file))

    def load(self, dir_path, custom={'f1':f1}):
        if not os.path.exists(dir_path):
            raise OSError('Directory \'{}\' not found.'.format(dir_path))
        else:
            aspectModel = load_model(os.path.join(dir_path, self.weight_file), custom_objects=custom)
        return aspectModel