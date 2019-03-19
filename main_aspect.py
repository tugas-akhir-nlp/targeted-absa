from Preprocessor import Preprocessor
from aspect.AspectCategorizer import AspectCategorizer

import time

if __name__ == '__main__':
    model = AspectCategorizer(
        normalize = False,
        lowercase = True,
        remove_punct = False,
        masking = False,
        embedding = False,
        trainable_embedding = True,
        pos_tag = True,
        dependency = True,
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

    x_train, y_train, x_test, y_test = model.preprocessor.get_all_input()

    model.train(
        x_train, 
        y_train,
        batch_size = 16,
        epochs = 30,
        verbose = 1,
        validation_split = 0.1,
        cross_validation = False,
        n_fold = 3,
        grid_search = False,
        callbacks = None
        )
    model.evaluate(x_train, y_train, x_test, y_test)
    model.evaluate_each_aspect(x_test, y_test)

    named_tuple = time.localtime()
    time_string = time.strftime("%B-%d-%Y_%H-%M-%S", named_tuple)
    model.save('aspect/model/{}'.format(time_string))

    # model.predict("Bensin nya irit banget nih tapi sayang kalo buat bepergian jauh mesinnya kurang kuat.")
    # model.predict("Pake mobil ini memang gak pernah kecewa, servis nya cepet sekali")
    print('\n')