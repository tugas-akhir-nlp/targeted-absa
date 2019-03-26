from Preprocessor import Preprocessor
from Preprocessor import MAX_LENGTH, EMBEDDING_SIZE, ASPECT_LIST
from metrics import f1

import tensorflow as tf
import numpy as np
import json
import os

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed, Lambda
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D


class AspectCategorizer():
    config_file = 'config.json'
    weight_file = 'model.h5'
    result_file = 'result.txt'

    def __init__ (
            self,
            module_name = 'aspect',
            train_file = None,
            test_file = None,
            normalize = False,
            lowercase = True,
            remove_punct = True,
            embedding = True,
            trainable_embedding = True,
            pos_tag = None,
            dependency = None,
            use_rnn = True,
            rnn_type = 'lstm',
            return_sequence = True,
            use_cnn = False,
            use_svm = False,
            use_stacked_svm = False,
            use_attention = False,
            n_neuron = 128,
            n_dense = 1,
            dropout = 0.5,
            regularizer = None,
            optimizer = 'adam',
            learning_rate = 0.001,
            weight_decay = 0):
        self.preprocessor  =  Preprocessor(
            module_name = module_name,
            train_file = train_file,
            test_file = test_file,
            normalize = normalize,
            lowercase = lowercase,
            remove_punct = remove_punct,
            embedding = embedding,
            pos_tag = pos_tag,
            dependency = dependency
        )        
        self.tokenizer = self.preprocessor.get_tokenized()
        self.aspects = ASPECT_LIST
        self.model = None
        self.history = None
        self.result = None
        self.module_name = 'aspect'

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
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.batch_size = None
        self.epochs = None
        self.verbose = None
        self.validation_split = None
        self.cross_validation = None
        self.n_fold = None
        self.grid_search = None
        self.callbacks = None

        self.score = list()
        self.result = {
            'pred' : None,
            'true' : None
        }

        print("Object has been created")

    def __get_config(self):
        keys = [
            'embedding',
            'trainable_embedding',
            'pos_tag',
            'dependency',
            'use_rnn',
            'rnn_type',
            'use_cnn',
            'use_svm',
            'use_stacked_svm',
            'use_attention',
            'n_neuron',
            'n_dense',
            'dropout',
            'regularizer',
            'optimizer',
            'learning_rate',
            'weight_decay',
            'batch_size',
            'epochs',
            'verbose',
            'validation_split',
            'cross_validation',
            'n_fold',
            'grid_search',
            'callbacks',
        ]
        return {k: getattr(self, k) for k in keys}

    def __build_model(self):
        print("Building the model...")
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
                _, pos_size = self.preprocessor.get_pos_dict()
                pos_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='pos_input')
                x2 = Lambda(
                    K.one_hot, 
                    arguments={'num_classes': pos_size}, 
                    output_shape=(MAX_LENGTH, pos_size)
                )(pos_input)
                x = keras.layers.concatenate([x, x2])

        else:
            new_embedding_size = EMBEDDING_SIZE
            if self.pos_tag is 'one_hot':
                new_embedding_size += 27
            if self.dependency is True:
                new_embedding_size += 2
            print('embedding size: ', new_embedding_size)
            main_input = Input(shape=(MAX_LENGTH, new_embedding_size), name='main_input')

        print("1. Input")
        
        if self.use_rnn is True:
            if self.embedding is True:
                if self.rnn_type is 'gru':
                    x = Bidirectional(GRU(self.n_neuron, return_sequences=True))(x)
                else:
                    x = Bidirectional(LSTM(self.n_neuron, return_sequences=True))(x)
            else:
                if self.rnn_type is 'gru':
                    x = Bidirectional(GRU(self.n_neuron, return_sequences=True))(main_input)
                else:
                    x = Bidirectional(LSTM(self.n_neuron, return_sequences=True))(main_input)
            x = GlobalMaxPool1D()(x)
            x = Dropout(self.dropout)(x)

        print("2. LSTM")

        if self.use_cnn is True:
            pass

        if self.n_dense is not 0:
            for i in range(self.n_dense):
                x = Dense(self.n_neuron, activation='relu')(x)
                x = Dropout(self.dropout)(x)

        print("3. Dense")

        out = Dense(len(self.aspects), activation='sigmoid')(x)

        print("4. Out")

        if self.pos_tag is 'embedding':
            model = Model([main_input, pos_input], out)
        else:
            model = Model(main_input, out)

        print("5. Model")

        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=[f1]
        )

        print("6. Done")

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

        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.validation_split = validation_split
        self.cross_validation = cross_validation
        self.n_fold = n_fold
        self.grid_search = grid_search
        self.callbacks = callbacks

        model = self.__build_model()

        print("Training...")

        if self.pos_tag is 'embedding':
            pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')
            x_input = [x_train, pos_train]
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
        x_name = ['Train-All', 'Test-All']

        print("======================= EVALUATION =======================")
        title = '{:10s} {:10s} {:10s} {:10s} {:10s}'.format('ASPECT', 'ACC', 'PREC', 'RECALL', 'F1')
        self.score.append(title)
        print(title)
        
        if self.validation_split > 0.0:
            print("Best epoch : ", np.argmax(self.history.history['val_f1'])+1)

        for i in range(2):
            if self.pos_tag is 'embedding':
                pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')
                pos_test = self.preprocessor.read_pos('resource/postag_test_auto.json')
                x_pos = [pos_train, pos_test]

                y_pred = self.model.predict([x[i], x_pos[i]])
            else:
                y_pred = self.model.predict(x[i])
            y_true = x[i+2]

            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_true)

            y_pred = (y_pred>0.5).astype(int)
            y_true = (y_true>0).astype(int)

            acc = accuracy_score(y_true.reshape([-1]), y_pred.reshape([-1]))
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')

            self.result = {
                'pred' : y_pred,
                'true' : y_true
            }

            score = '{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} '.format(x_name[i], acc, precision, recall, f1)
            print(score)
            self.score.append(score)
        

    def evaluate_each_aspect(self, x_test, y_test):
        if self.pos_tag is 'embedding':
            pos_test = self.preprocessor.read_pos('resource/postag_test_auto.json')
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
            score = '{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}'.format(self.aspects[i], acc, precision, recall, f1)
            print(score)
            self.score.append(score)

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
        y = self.__get_config()
        with open(os.path.join(dir_path, self.config_file), 'w') as f:
            f.write(json.dumps(y, indent=4, sort_keys=True))

        review_test = self.preprocessor.read_data_for_aspect('aspect/data/aspect_test.json')

        with open(os.path.join(dir_path, self.result_file), 'w') as f:
            f.write("======================= EVALUATION =======================\n")
            for score in self.score:
                f.write(score + "\n")
            f.write("\n")
            f.write("======================= PREDICTION =======================\n")
            for i, pred in enumerate(self.result['pred']):
                f.write(str(i) + "\n")
                f.write(review_test[i]+ "\n")
                temp = list()
                true = list()
                for j, asp in enumerate(pred):
                    if asp == 1:
                        temp.append(self.aspects[j])
                    if self.result['true'][i][j] == 1:
                        true.append(self.aspects[j])
                f.write("TRUE: " + json.dumps(true) + "\n")
                f.write("PRED: " + json.dumps(temp) + "\n")

    def load(self, dir_path, custom={'f1':f1}):
        if not os.path.exists(dir_path):
            raise OSError('Directory \'{}\' not found.'.format(dir_path))
        else:
            aspectModel = load_model(os.path.join(dir_path, self.weight_file), custom_objects=custom)
            self.model = aspectModel
        return self.model