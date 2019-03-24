from Preprocessor import Preprocessor
from Preprocessor import MAX_LENGTH, EMBEDDING_SIZE, ASPECT_LIST

import tensorflow as tf
import numpy as np
import json
import os

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from keras import backend as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D

class SentimentClassifier():
    config_file = 'config.json'
    weight_file = 'model.h5'
    result_file = 'result.txt'

    def __init__ (
            self,
            module_name = 'sentiment',
            train_file = None,
            test_file = None,
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
            masking = masking,
            embedding = embedding,
            pos_tag = pos_tag,
            dependency = dependency
        )        
        self.tokenizer = self.preprocessor.get_tokenized()
        self.aspects = ASPECT_LIST
        self.model = None
        self.history = None
        self.result = None
        self.module_name = 'sentiment'

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
            new_embedding_size = EMBEDDING_SIZE + 6
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

        out = Dense(2, activation='softmax')(x)

        print("4. Out")

        if self.pos_tag is 'embedding':
            model = Model([main_input, pos_input], out)
        else:
            model = Model(main_input, out)

        print("5. Model")

        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=self.optimizer,
            metrics=['acc']
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

        history = model.fit(
            x = x_train, 
            y = y_train, 
            batch_size = batch_size,
            epochs = epochs, 
            verbose = verbose,
            validation_split = validation_split,
            callbacks = callbacks
        )

        self.model = model
        self.history = history

    def change_to_multilabel(self, aspects, sentiments):
        multilabel = list()
        idx = 0
        for data in aspects:
            temp = list()
            for aspect in data:
                if aspect == 1:
                    if sentiments[idx] == 0:
                        temp.append(2)
                    else:
                        temp.append(1)
                    idx += 1
                else:
                    temp.append(0)
            multilabel.append(temp)

        return np.array(multilabel)

    def evaluate(self, x_train, y_train, x_test, y_test):
        x = [x_train, x_test, y_train, y_test]
        # x_pos = [pos_train, pos_test]
        x_name = ['Train-All', 'Test-All']

        _, y_train_aspect, _, y_test_aspect = self.preprocessor.get_all_input_aspect()

        x_aspect = [y_train_aspect, y_test_aspect]

        print("======================= EVALUATION =======================")
        print('{:10s} {:10s} {:10s} {:10s} {:10s}'.format('ASPECT', 'ACC', 'PREC', 'RECALL', 'F1'))

        self.evaluate_all(x_train, x_test, y_train, y_test, x_aspect)
        self.evaluate_each_aspect(x_test, y_test, y_test_aspect)

    def evaluate_all(self, x_train, x_test, y_train, y_test, x_aspect):
        x = [x_train, x_test, y_train, y_test]
        # x_pos = [pos_train, pos_test]
        x_name = ['Train-All', 'Test-All']        

        for i in range(2):
            if self.pos_tag is 'embedding':
                y_pred = self.model.predict([x[i], x_pos[i]])
            else:
                y_pred = self.model.predict(x[i])
            y_true = x[i+2]

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)

            acc = accuracy_score(y_true, y_pred)

            if i == 1:
                self.result = {
                    'pred' : y_pred,
                    'true' : y_true
                }

            y_pred = self.change_to_multilabel(x_aspect[i], y_pred)
            y_true = self.change_to_multilabel(x_aspect[i], y_true)

            pred_reshape = np.reshape(y_pred, -1)
            true_reshape = np.reshape(y_true, -1)

            precision = precision_score(true_reshape, pred_reshape, average='macro')
            recall = recall_score(true_reshape, pred_reshape, average='macro')
            f1 = f1_score(true_reshape, pred_reshape, average='macro')

            print('{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} '.format(x_name[i], acc, precision, recall, f1))


    def evaluate_each_aspect(self, x_test, y_test, y_test_aspect):
        y_pred = self.change_to_multilabel(y_test_aspect, self.result['pred'])
        y_true = self.change_to_multilabel(y_test_aspect, self.result['true'])

        true_transpose = y_true.transpose()
        pred_transpose = y_pred.transpose()

        class_names = [0,1,2]
        f1pos = list()
        f1neg = list()
        f1avg = list()

        cnf_matrix = list()
        for i in range(len(self.aspects)):
            cnf_matrix.append(
                confusion_matrix(
                    true_transpose[i], 
                    pred_transpose[i], 
                    labels=class_names)
                )
            np.set_printoptions(precision=2)

        for i in range(len(self.aspects)):
            precpos = cnf_matrix[i][1][1]/(cnf_matrix[i][0][1]+cnf_matrix[i][1][1]+cnf_matrix[i][2][1])
            recpos = cnf_matrix[i][1][1]/(cnf_matrix[i][1][0]+cnf_matrix[i][1][1]+cnf_matrix[i][1][2])
            f1p = 2*(precpos*recpos)/(precpos+recpos)
            f1pos.append(f1p)

            precneg = cnf_matrix[i][2][2]/(cnf_matrix[i][0][2]+cnf_matrix[i][1][2]+cnf_matrix[i][2][2])
            recneg = cnf_matrix[i][2][2]/(cnf_matrix[i][2][0]+cnf_matrix[i][2][1]+cnf_matrix[i][2][2])
            f1n = 2*(precneg*recneg)/(precneg+recneg)
            f1neg.append(f1n) 

            precw = ((precpos*cnf_matrix[i][1][1]) + (precneg*cnf_matrix[i][2][2]))/(cnf_matrix[i][1][1] + cnf_matrix[i][2][2])
            recw = ((recpos*cnf_matrix[i][1][1]) + (recneg*cnf_matrix[i][2][2]))/(cnf_matrix[i][1][1] + cnf_matrix[i][2][2])
            f1w = 2*(precw*recw)/(precw+recw)
            f1avg.append(f1w)
            
            print('{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} '.format(self.aspects[i], class_names[0], precw, recw, f1w))

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
        y = self.__get_config()
        print(y)
        with open(os.path.join(dir_path, self.config_file), 'w') as f:
            f.write(json.dumps(y, indent=4, sort_keys=True))

        review_test = self.preprocessor.read_data_for_sentiment('sentiment/data/sentiment_test.json')

        # with open(os.path.join(dir_path, self.result_file), 'w') as f:
        #     for i, pred in enumerate(self.result['pred']):
        #         f.write(review_test[i])
        #         temp = list()
        #         true = list()
        #         for j, asp in enumerate(pred):
        #             if asp == 1:
        #                 temp.append(self.aspects[j])
        #             if self.result['true'] == 1:
        #                 true.append(self.aspects[j])
        #         f.write(json.dump(temp) + "\n")
        #         f.write(json.dump(true) + "\n")

    def load(self, dir_path):
        if not os.path.exists(dir_path):
            raise OSError('Directory \'{}\' not found.'.format(dir_path))
        else:
            self.model = load_model(os.path.join(dir_path, self.weight_file))
        return self.model