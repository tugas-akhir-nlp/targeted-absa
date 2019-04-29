from Preprocessor import Preprocessor
from Preprocessor import MAX_LENGTH, EMBEDDING_SIZE, ASPECT_LIST
from metrics import f1

import tensorflow as tf
import numpy as np
import json
import os

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras_pos_embd import PositionEmbedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed, Lambda
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, AveragePooling1D, GlobalAvgPool1D


class AspectCategorizer():
    config_file = 'config.json'
    weight_file = 'model.h5'
    result_file = 'result.txt'
    pred_file = 'pred.txt'

    def __init__ (
            self,
            module_name = 'aspect',
            train_file = None,
            test_file = None,
            lowercase = True,
            remove_punct = True,
            embedding = True,
            trainable_embedding = True,
            pos_tag = None,
            dependency = None,
            use_entity = True,
            postion_embd = True,
            mask_entity = False,
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
            position_embd = postion_embd,
            mask_entity = mask_entity,
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
        self.use_entity = use_entity
        self.position_embd = postion_embd
        self.mask_entity = mask_entity
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
        self.validation_data = None
        self.data_val = None
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
            'cross_validation',
            'n_fold',
            'grid_search',
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

            if self.position_embd == True:
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

            if self.pos_tag is 'embedding':
                _, pos_size = self.preprocessor.get_pos_dict()
                pos_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='pos_input')
                x3 = Lambda(
                    K.one_hot, 
                    arguments={'num_classes': pos_size}, 
                    output_shape=(MAX_LENGTH, pos_size)
                )(pos_input)
                x = keras.layers.concatenate([x, x3])

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
            # x = GlobalAvgPool1D()(x)
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

        x_input = list()
        x_input.append(main_input)

        if self.position_embd == True:
            x_input.append(position_input)
        if self.pos_tag is 'embedding':
            x_input.append(pos_input)
        
        model = Model(x_input, out)

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
        validation_data = False,
        cross_validation = False,
        n_fold = 3,
        grid_search = False,
        callbacks = None):

        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.validation_data = validation_data
        self.cross_validation = cross_validation
        self.n_fold = n_fold
        self.grid_search = grid_search
        self.callbacks = callbacks

        model = self.__build_model()

        print("Training...")

        x_input = list()

        if self.validation_data:
            input_val = list()
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=70)
            input_val.append(x_val)
    
        x_input.append(x_train)

        if self.position_embd:
            if self.mask_entity:
                position_train = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.train_file)  
            else:       
                position_train = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.train_file) 

            if self.validation_data:
                position_train, position_val = train_test_split(position_train, test_size=0.1, random_state=70)    
                input_val.append(position_val)
            x_input.append(position_train)

        if self.pos_tag == 'embedding':
            pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')   
            if self.validation_data:
                pos_train, pos_val = train_test_split(pos_train, test_size=0.1, random_state=70)
                input_val.append(pos_val)
            x_input.append(pos_train)

        history = model.fit(
            x = x_input, 
            y = y_train, 
            batch_size = batch_size,
            epochs = epochs, 
            verbose = verbose,
            validation_data = [input_val, y_val],
            callbacks = callbacks
        )

        self.data_val = [x_val, y_val]
        self.model = model
        self.history = history

    def evaluate(self, x_train, y_train, x_test, y_test):
        if self.validation_data:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=70)
            x = [x_train, x_val, y_train, y_val]
            x_name = ['Train-All', 'Val-All']
        else:
            x = [x_train, x_test, y_train, y_test]
            x_name = ['Train-All', 'Test-All']

        if self.pos_tag is 'embedding':
            pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')
            pos_test = self.preprocessor.read_pos('resource/postag_test_auto.json')

            if self.validation_data:
                pos_train, pos_val = train_test_split(pos_train, test_size=0.1, random_state=70)
                x_pos = [pos_train, pos_val]
            else:
                x_pos = [pos_train, pos_test]

        if self.position_embd == True:
            if self.mask_entity:
                position_train = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.train_file) 
                position_test = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.test_file) 
            else:
                position_train = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.train_file) 
                position_test = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.test_file) 
            
            if self.validation_data:
                position_train, position_val = train_test_split(position_train, test_size=0.1, random_state=70)
                x_position = [position_train, position_val]
            else:
                x_position = [position_train, position_test]


        print("======================= EVALUATION =======================")
        title = '{:10s} {:10s} {:10s} {:10s} {:10s}'.format('ASPECT', 'ACC', 'PREC', 'RECALL', 'F1')
        self.score.append(title)
        print(title)
        

        for i in range(2):
            if self.position_embd == True and self.pos_tag == 'embedding':            
                y_pred = self.model.predict([x[i], x_position[i], x_pos[i]])
            elif self.position_embd == True and self.pos_tag != 'embedding':
                y_pred = self.model.predict([x[i], x_position[i]])
            elif self.position_embd == False and self.pos_tag == 'embedding':
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
        
        if self.validation_data:
            print("Best epoch : ", np.argmax(self.history.history['val_f1'])+1)
        
        return precision, recall, f1

    def evaluate_each_aspect(self, x_test, y_test):
        x_input = list()

        if self.validation_data:
            x_input.append(self.data_val[0])
            y_test = self.data_val[1]
        else:
            x_input.append(x_test)

        if self.position_embd:
            if self.mask_entity:
                position_test = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.test_file)
            else:         
                position_test = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.test_file)         

            if self.validation_data:
                position_train = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.train_file)
                position_train, position_val = train_test_split(position_train, test_size=0.1, random_state=70)
                position_test = position_val
            x_input.append(position_test)

        if self.pos_tag == 'embedding':
            if self.validation_data:
                pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')    
                pos_train, pos_val = train_test_split(pos_train, test_size=0.1, random_state=70)
                pos_test = pos_val
            else:   
                pos_test = self.preprocessor.read_pos('resource/postag_test_auto.json')   
            
            x_input.append(pos_test)
        
        y_pred = self.model.predict(x_input)
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
        with open(os.path.join(dir_path, self.result_file), 'w') as f:
            f.write("======================= EVALUATION =======================\n")
            for score in self.score:
                f.write(score + "\n")
            f.write("\n")

        if not self.validation_data:
            review_test = self.preprocessor.read_data_for_aspect('data/entity_test.json')
            entities = self.preprocessor.get_entities('data/entity_test.json')

            with open(os.path.join(dir_path, self.pred_file), 'w') as f:
                for pred in self.result['pred']:
                    f.write(str(pred) + "\n")

            with open(os.path.join(dir_path, self.result_file), 'w') as f:
                f.write("======================= EVALUATION =======================\n")
                for score in self.score:
                    f.write(score + "\n")
                f.write("\n")
                f.write("==================== WRONG PREDICTION ====================\n")
                for i, pred in enumerate(self.result['pred']):
                    temp = list()
                    true = list()
                    if list(pred) != list(self.result['true'][i]):
                        f.write(str(i) + "\n")
                        f.write(review_test[i]+ "\n")
                        for j, asp in enumerate(pred):
                            if asp == 1:
                                temp.append(self.aspects[j])
                            if self.result['true'][i][j] == 1:
                                true.append(self.aspects[j])
                        f.write("TRUE: " + entities[i] + " - " + json.dumps(true) + "\n")
                        f.write("PRED: " + entities[i] + " - " + json.dumps(temp) + "\n")

                f.write("\n==================== TRUE PREDICTION ====================\n")
                for i, pred in enumerate(self.result['pred']):
                    temp = list()
                    true = list()
                    if list(pred) == list(self.result['true'][i]):
                        f.write(str(i) + "\n")
                        f.write(review_test[i]+ "\n")
                        for j, asp in enumerate(pred):
                            if asp == 1:
                                temp.append(self.aspects[j])
                            if self.result['true'][i][j] == 1:
                                true.append(self.aspects[j])
                        f.write("TRUE: " + entities[i] + " - " + json.dumps(true) + "\n")
                        f.write("PRED: " + entities[i] + " - " + json.dumps(temp) + "\n")

    def load(self, dir_path, custom={'f1':f1, 'PositionEmbedding':PositionEmbedding}):
        if not os.path.exists(dir_path):
            raise OSError('Directory \'{}\' not found.'.format(dir_path))
        else:
            aspectModel = load_model(os.path.join(dir_path, self.weight_file), custom_objects=custom)
            self.model = aspectModel
        return self.model