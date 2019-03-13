from aspect.AspectCategorizer import AspectCategorizer
from Preprocessor import Preprocessor

if __name__ == '__main__':
    preprocessor = Preprocessor()
    x_train, y_train, x_test, y_test = preprocessor.get_encoded_input()
    model = AspectCategorizer(
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
        n_neuron = 256,
        n_dense = 1,
        dropout = 0.5,
        optimizer = 'adam'
        )
    model.train(
        x_train, 
        y_train,
        batch_size = 16,
        epochs = 10,
        verbose = 1,
        validation_split = 0.0,
        cross_validation = False,
        n_fold = 3,
        grid_search = False,
        callbacks = None
        )
    model.evaluate(x_train, y_train, x_test, y_test)
    model.evaluate_each_aspect(x_test, y_test)
    model.predict("Bensin nya irit banget nih tapi sayang kalo buat bepergian jauh mesinnya kurang kuat.")
    print('\n')