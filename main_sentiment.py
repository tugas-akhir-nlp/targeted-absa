from Preprocessor import Preprocessor
from sentiment.SentimentClassifier import SentimentClassifier

import time
import tensorflow as tf
from keras import backend as k

if __name__ == '__main__':
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    k.tensorflow_backend.set_session(tf.Session(config=config))

    model = SentimentClassifier(
        train_file = 'sentiment/data/sentiment_train.json',
        normalize = False,
        lowercase = True,
        remove_punct = True,
        masking = False,
        embedding = False,
        trainable_embedding = True,
        pos_tag = False,
        dependency = False,
        use_rnn = True,
        rnn_type = 'lstm',
        use_cnn = False,
        use_svm = False,
        use_stacked_svm = False,
        use_attention = False,
        n_neuron = 256,
        n_dense = 1,
        dropout = 0.5,
        regularizer = None,
        optimizer = 'adam'
        )

    x_train, y_train, x_test, y_test = model.preprocessor.get_all_input_sentiment()

    model.train(
        x_train, 
        y_train,
        batch_size = 32,
        epochs = 5,
        verbose = 1,
        validation_split = 0.0,
        cross_validation = False,
        n_fold = 3,
        grid_search = False,
        callbacks = None
        )
    model.evaluate(x_train, y_train, x_test, y_test)

    named_tuple = time.localtime()
    time_string = time.strftime("%B-%d-%Y_%H-%M-%S", named_tuple)
    model.save('sentiment/model/{}'.format(time_string))

    # model.load('sentiment/model/March-18-2019_23-36-29')
    # model.evaluate(x_train, y_train, x_test, y_test)

    # model.predict("Bensin nya irit banget nih tapi sayang kalo buat bepergian jauh mesinnya kurang kuat.")
    # model.predict("Pake mobil ini memang gak pernah kecewa, servis nya cepet sekali")
    print('\n')