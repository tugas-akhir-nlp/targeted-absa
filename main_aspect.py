from Preprocessor import Preprocessor
from aspect.AspectCategorizer import AspectCategorizer

import time
import tensorflow as tf
from keras import backend as k
from sklearn.model_selection import train_test_split

from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    k.tensorflow_backend.set_session(tf.Session(config=config))

    param_grid = {
    'epochs' : [10,15],
    'batch_size': [16,32], 
    'dropout': [0.3,0.5,0.7], 
    'neuron_unit': [128,256]
    }

    grid = ParameterGrid(param_grid)
    with open('result_asp_gs.txt', 'w') as f:
        for i, params in enumerate(grid):
            print(i)
            f.write(str(i) + "\n")
            print(params)
            model = AspectCategorizer(
                train_file = 'data/entity_train.json',
                test_file= 'data/entity_test.json',
                lowercase = True,
                remove_punct = True,
                embedding = True,
                trainable_embedding = True,
                pos_tag = 'embedding',
                dependency = False,
                use_entity = True,
                postion_embd = True,
                mask_entity = False,    
                use_rnn = True,
                rnn_type = 'lstm',
                use_cnn = False,
                use_svm = False,
                use_stacked_svm = False,
                use_attention = False,
                n_neuron = params['neuron_unit'],
                n_dense = 1,
                dropout = params['dropout'],
                regularizer = None,
                optimizer = 'adam'
                )

            x_train, y_train, x_test, y_test = model.preprocessor.get_all_input_aspect()

            model.train(
                x_train, 
                y_train,
                batch_size = params['batch_size'],
                epochs = params['epochs'],
                verbose = 1,
                validation_data = True,
                cross_validation = False,
                n_fold = 3,
                grid_search = False,
                callbacks = None
                )

            # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=70)

            # model.load('aspect/model/April-18-2019_16-54-37')
            print(params)
            prec, rec, f1 = model.evaluate(x_train, y_train, x_test, y_test)
            f.write(str(params) + ' : ' + str(prec) + ' ' + str(rec) + ' ' + str(f1) + "\n")

    # model.evaluate_each_aspect(x_test, y_test)

    # named_tuple = time.localtime()
    # time_string = time.strftime("%B-%d-%Y_%H-%M-%S", named_tuple)
    # model.save('aspect/model/{}'.format(time_string))

    # model.predict("Bensin nya irit banget nih tapi sayang kalo buat bepergian jauh mesinnya kurang kuat.")
    # model.predict("Pake mobil ini memang gak pernah kecewa, servis nya cepet sekali")
    print('\n')