from aspect.AspectCategorizer import AspectCategorizer
from Preprocessor import Preprocessor

if __name__ == '__main__':
    preprocessor = Preprocessor()
    x_train, y_train, x_test, y_test = preprocessor.get_encoded_input()
    model = AspectCategorizer()
    model.train(x_train, y_train)
    model.evaluate(x_train, y_train, x_test, y_test)