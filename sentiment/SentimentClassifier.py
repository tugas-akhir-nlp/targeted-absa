from Preprocessor import Preprocessor
from Preprocessor import MAX_LENGTH, EMBEDDING_SIZE, ASPECT_LIST

import tensorflow as tf
import numpy as np
import json
import os

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

import keras
from keras import backend as K
from keras_pos_embd import PositionEmbedding
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
            lowercase = True,
            remove_punct = True,
            embedding = True,
            trainable_embedding = True,
            pos_tag = None,
            dependency = None,
            use_entity = True,
            use_lexicon = True,
            use_op_target = True,
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
            lowercase = lowercase,
            remove_punct = remove_punct,
            embedding = embedding,
            pos_tag = pos_tag,
            dependency = dependency,
            use_entity = use_entity,
            use_lexicon = use_lexicon,
            use_op_target = use_op_target
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
        self.use_entity = use_entity
        self.use_lexicon = use_lexicon
        self.use_op_target = use_op_target
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
            'true' : None,
            'join_pred' : None,
            'join_true' : None,
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

            aspect_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='aspect_input')
            x2 = keras.layers.Lambda(
                K.one_hot, 
                arguments={'num_classes': len(ASPECT_LIST)}, 
                output_shape=(MAX_LENGTH, len(ASPECT_LIST))
            )(aspect_input)
            x = keras.layers.concatenate([x, x2])

            if self.use_entity == True:
                weights = np.random.random((201, 50))
                position_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='position_input')
                x2 = PositionEmbedding(
                    input_shape=(MAX_LENGTH,),
                    input_dim=100,    
                    output_dim=50,     
                    weights=[weights],
                    mode=PositionEmbedding.MODE_EXPAND,
                    name='position_embedding',
                )(position_input)
                x = keras.layers.concatenate([x, x2])

            if self.use_lexicon == True:
                lex_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='lex_input')
                x3 = keras.layers.Lambda(
                    K.one_hot, 
                    arguments={'num_classes': 3}, 
                    output_shape=(MAX_LENGTH, 3)
                )(lex_input)
                x = keras.layers.concatenate([x, x3])

            if self.pos_tag is 'embedding':
                _, pos_size = self.preprocessor.get_pos_dict()
                pos_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='pos_input')
                x4 = keras.layers.Lambda(
                    K.one_hot, 
                    arguments={'num_classes': pos_size}, 
                    output_shape=(MAX_LENGTH, pos_size)
                )(pos_input)
                x = keras.layers.concatenate([x, x4])

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

        x_input = list()
        x_input.append(main_input)
        x_input.append(aspect_input)

        if self.use_entity == True:
            x_input.append(position_input)
        if self.use_lexicon == True:
            x_input.append(lex_input)
        if self.pos_tag is 'embedding':
            x_input.append(pos_input)
        
        model = Model(x_input, out)

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

        x_input = list()
        x_input.append(x_train)

        _, aspect_train = self.preprocessor.read_sentiment(self.preprocessor.train_file, x_train)
        x_input.append(aspect_train)

        if self.use_entity == True:
            position_train = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.train_file, 'data/entity_train.json')         
            x_input.append(position_train)
        if self.use_lexicon == True:
            posneg_train = self.preprocessor.get_sentiment_lexicons('data/entity_train.json')
            x_input.append(posneg_train)
        if self.pos_tag == 'embedding':
            pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')   
            x_input.append(pos_train)

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
        x_name = ['Train-All', 'Test-All']
        _, y_train_aspect, _, y_test_aspect = self.preprocessor.get_all_input_aspect()
        x_aspect = [y_train_aspect, y_test_aspect]

        print("======================= EVALUATION =======================")
        title = '{:10s} {:10s} {:10s} {:10s} {:10s}'.format('ASPECT', 'ACC', 'PREC', 'RECALL', 'F1')
        self.score.append(title)
        print(title)

        self.evaluate_all(x_train, x_test, y_train, y_test, x_aspect)
        self.evaluate_each_aspect(x_test, y_test, y_test_aspect)

    def evaluate_all(self, x_train, x_test, y_train, y_test, x_aspect):
        x = [x_train, x_test, y_train, y_test]        
        x_name = ['Train-All', 'Test-All'] 

        _, aspect_train = self.preprocessor.read_sentiment(self.preprocessor.train_file, x_train)
        _, aspect_test = self.preprocessor.read_sentiment(self.preprocessor.test_file, x_test)

        x_asp = [aspect_train, aspect_test]

        if self.use_entity == True:
            position_train = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.train_file, 'data/entity_train.json')
            position_test = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.test_file, 'data/entity_test.json')
            x_position = [position_train, position_test] 

        if self.use_lexicon == True:
            posneg_train = self.preprocessor.get_sentiment_lexicons('data/entity_train.json')
            posneg_test = self.preprocessor.get_sentiment_lexicons('data/entity_test.json')
            x_lex = [posneg_train, posneg_test]

        if self.pos_tag is 'embedding':
            pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')
            pos_test = self.preprocessor.read_pos('resource/postag_test_auto.json')
            x_pos = [pos_train, pos_test]      

        for i in range(2):
            x_input = list()
            x_input.append(x[i])
            x_input.append(x_asp[i])

            if self.use_entity:
                x_input.append(x_position[i])
            if self.use_lexicon:
                x_input.append(x_lex[i])

            y_pred = self.model.predict(x_input)
            y_true = x[i+2]

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)

            y_pred_multilabel = self.change_to_multilabel(x_aspect[i], y_pred)
            y_true_multilabel = self.change_to_multilabel(x_aspect[i], y_true)

            pred_reshape = np.reshape(y_pred_multilabel, -1)
            true_reshape = np.reshape(y_true_multilabel, -1)

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(true_reshape, pred_reshape, average='macro')
            recall = recall_score(true_reshape, pred_reshape, average='macro')
            f1 = f1_score(true_reshape, pred_reshape, average='macro')

            if i == 1:
                self.result = {
                    'pred' : y_pred,
                    'true' : y_true,
                    'join_pred' : y_pred_multilabel,
                    'join_true' : y_true_multilabel
                }

            score = '{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} '.format(x_name[i], acc, precision, recall, f1)
            print(score)
            self.score.append(score)


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
            
            score = '{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}'.format(self.aspects[i], class_names[0], precw, recw, f1w)
            print(score)
            self.score.append(score)

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
        with open(os.path.join(dir_path, self.config_file), 'w') as f:
            f.write(json.dumps(y, indent=4, sort_keys=True))

        review_test = self.preprocessor.read_data_for_sentiment('data/entity_test.json')
        entities = self.preprocessor.get_entities('data/entity_test.json')

        with open(os.path.join(dir_path, self.result_file), 'w') as f:
            f.write("======================= EVALUATION =======================\n")
            for score in self.score:
                f.write(score + "\n")
            f.write("\n")
            f.write("======================= PREDICTION =======================\n")
            idx = 0
            for i, pred in enumerate(self.result['join_pred']):
                f.write(str(i) + "\n")
                for j, asp in enumerate(pred):
                    if asp != 0:
                        f.write(review_test[idx] + "\n")
                        idx += 1
                        if asp == 1:
                            if self.result['join_true'][i][j] == 1:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - positive\n")
                            else:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - negative\n")
                            f.write("PRED: "+ entities[i] + " - " + self.aspects[j] + " - positive\n")
                        elif asp == 2:
                            if self.result['join_true'][i][j] == 1:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - positive\n")
                            else:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - negative\n")
                            f.write("PRED: "+ entities[i] + " - " + self.aspects[j] + " - negative\n")
                    
    def load(self, dir_path):
        if not os.path.exists(dir_path):
            raise OSError('Directory \'{}\' not found.'.format(dir_path))
        else:
            self.model = load_model(os.path.join(dir_path, self.weight_file))
        return self.model