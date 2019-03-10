import Preprocessor
from Preprocessor import MAX_LENGTH, EMBEDDING_SIZE, ASPECTS

import tensorflow as tf

from sklearn.metrics import accuracy_score

from keras import backend as k
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D

class SentimentClassifier():
    def __init__ (
            self, 
            normalize = True,
            lowercase = True,
            remove_punct = True,
            max_length = max_lenth,
            masking = True,
            embedding = True,
            embedding_size = embedding_size,
            trainable_embedding = True,
            pos_tag = None,
            use_rnn = True,
            rnn_type = 'lstm',
            use_cnn = False,
            use_attention = False,
            n_neuron = 128,
            n_dense = 1,
            dropout = 0.5,
            optimizer = 'adam'):
        self.preprocessor  =  Preprocessor(
            normalize = normalize,
            lowercase = lowercase,
            remove_punct = remove_punct,
            max_length = max_length,
            masking = masking
        )
        self.tokenizer = self.preprocessor.get_tokenized()
        self.aspects = aspects
        self.model = None
        self.history = None

        self.embedding = embedding
        self.embedding_size = embedding_size
        self.trainable_embedding = trainable_embedding
        self.pos_tag = pos_tag
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.use_cnn = use_cnn
        self.use_attention = use_attention
        self.n_neuron = n_neuron
        self.n_dense = n_dense
        self.dropout = dropout
        self.optimizer = optimizer

    def __build_model(self):
        vocab_size = self.preprocessor.get_vocab_size()

        if self.embedding:
            embedding_matrix = self.preprocessor.get_embedding_matrix(self.tokenizer)
            main_input = Input(shape=(self.max_length,), dtype='int32', name='main_input')
            x = Embedding(
                output_dim=self.embedding_size, 
                input_dim=vocab_size, 
                input_length=self.max_length,
                weights=[embedding_matrix],
                trainable=self.trainable_embedding,
            )(main_input)

            if self.pos_tag is 'embedding':
                pos_matrix = self.preprocessor.get_pos_matrix(review)
                pos_input = Input(shape=(self.max_length,), dtype='int32', name='pos_input')
                x2 = Embedding(
                    output_dim=30, 
                    input_dim=pos_size, 
                    input_length=self.max_length,
                    weights=[pos_matrix],
                    trainable=self.trainable_embedding,
                )(pos_input)
                x = keras.layers.concatenate([x, x2])

        else:
            embedded_input = self.preprocessor.get_embedded_input(review)

        if self.pos_tag is 'one_hot':
            encoded_pos = self.preprocessor.get_encoded_pos(review)

        if self.use_rnn is True:
            if rnn_type is 'gru':
                x = Bidirectional(GRU(self.n_neuron, return_sequences=True))(x)
            else:
                x = Bidirectional(LSTM(self.n_neuron, return_sequences=True))(x)
            x = GlobalMaxPool1D()(x)
            x = Dropout(self.dropout)(x)

        if self.use_cnn is True:
            continue

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
            metrics='acc'
        )

        return model

    def train(
        self, 
        x_train, 
        y_train,
        batch_size = 16,
        epochs = 10,
        verbose = 1,
        validation_split = 0.0,
        cross_validation = False,
        n_fold = 3,
        callbacks = None
        ):

        model = self.__build_model()

        if self.pos_tag is 'embedding':
            x_input = [x_train, pos_tag]
        else: 
            x_input = x_train

        history = self.model.fit(
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

    def evaluate(self, x_train, x_test, y_train, y_test):
        x = [x_train, x_test, y_train, y_test]
        x_pos = [pos_train, pos_test]
        x_name = ['TRAIN', 'TEST']

        print("=============== EVALUATION ===============")

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

            print(x_name[i])
            print('Akurasi : ', acc, '\n')

    def evaluate_each_aspect(self, x_test, y_test):
        print("=============== EVALUATION FOR EACH ASPECT ===============")

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
            print(self.aspects[i], acc)

    def predict(self, new):
        encoded_new = self.tokenizer.texts_to_sequences([new.lower()])
        input_new = pad_sequences(encoded_new, maxlen=self.max_length, padding='post')
        pred = self.model.predict(input_new)

        label = list()
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                if (pred[i][j] > 0.5):
                    label.append(aspects[j])

        return label

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        sentimentModel = load_model(file_path)
        return sentimentModel