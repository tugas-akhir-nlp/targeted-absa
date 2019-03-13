import json
import string
import numpy as np

from gensim.models import Word2Vec

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 100
EMBEDDING_SIZE = 500
ASPECTS = [
    'others',
    'machine',
    'part',
    'price',
    'service',
    'fuel'
    ]

class Preprocessor():
    def __init__(
            self,
            normalize=True,
            lowercase=True,
            remove_punct=True,
            masking=True):
        self.normalize = normalize
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.masking = masking

        self.punctuations = set(string.punctuation)

    def __normalize(self, text):
        if self.normalize:
            normalized_text = text
        return normalized_text

    def __lower(self, text):
        return text.lower() if self.lowercase else text

    def __remove_punct(self, text):
        if self.remove_punct:
            text_splitted = text.split()
            result = list()
            for text in text_splitted:
                result.append(''.join(ch for ch in text if ch not in self.punctuations))
        return ' '.join(result)

    def __mask_entity(self, text):
        return text

    def __load_embedding(self):
        with open('resource/w2v_path.txt') as file:
            word2vec_path = file.readlines()[0]            
        w2v = Word2Vec.load(word2vec_path)
        return w2v

    def __load_json(self):
        with open('aspect/data/aspect_train.json') as f:
            json_train = json.load(f)

        with open('aspect/data/aspect_test.json') as f:
            json_test = json.load(f)

        return json_train, json_test

    def __load_pos(self):
        with open('resource/postag_train_auto.json') as f:
            pos_train = json.load(f)

        with open('resource/postag_test_auto.json') as f:
            pos_test = json.load(f)

        return pos_train, pos_test

    def __load_pos_dict(self):
        with open('resource/pos_dict.json') as f:
            pos_dict = json.load(f)
        return pos_dict

    def __read_data(self, json_data):
        review = list()
        for data in json_data:
            temp = self.__lower(data['text'])
            temp = self.__remove_punct(temp)
            review.append(temp)

        return review

    def __read_label(self, json_data):
        label = list()
        for i in range(len(json_data)):
            temp = np.zeros(len(ASPECTS), dtype=int)
            for aspect in range(len(json_data[i]['aspects'])):
                if json_data[i]['aspects'] != []:
                    for j in range(len(ASPECTS)):
                        if ASPECTS[j] in json_data[i]['aspects'][aspect]:
                            temp[j] = 1
            label.append(temp)

        return np.array(label)
    
    def __read_pos(self, json_data):
        pos = list()
        pos_dict = self.__load_pos_dict()
        for data in json_data:
            temp = np.zeros(MAX_LENGTH, dtype=int)
            idx = 0
            for sentence in data['sentences']:
                for i, token in sentence['tokens']:
                    if i >= MAX_LENGTH:
                        continue
                    temp[idx] = pos_dict[token['pos_tag']]
                    idx += 1
            pos.append(temp)
        return pos

    def get_tokenized(self):
        json_train, _ = self.__load_json()
        review = self.__read_data(json_train)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(review)
        return tokenizer

    def get_vocab_size(self, tokenizer):
        return len(tokenizer.word_index) + 1

    def get_encoded_input(self):
        tokenizer = self.get_tokenized()
        json_train, json_test = self.__load_json()

        review = self.__read_data(json_train)
        review_test = self.__read_data(json_test)

        encoded_data = tokenizer.texts_to_sequences(review)
        encoded_data_test = tokenizer.texts_to_sequences(review_test)

        x_train = pad_sequences(encoded_data, maxlen=MAX_LENGTH, padding='post')
        x_test = pad_sequences(encoded_data_test, maxlen=MAX_LENGTH, padding='post')

        y_train = self.__read_label(json_train)
        y_test = self.__read_label(json_test)

        return x_train, y_train, x_test, y_test

    def get_embedding_matrix(self, tokenizer):
        w2v = self.__load_embedding()
        words = list(w2v.wv.vocab)
        embeddings_index = dict()

        for word in words:
            coefs = w2v[word]
            embeddings_index[word] = coefs

        vocab_size = self.get_vocab_size(tokenizer)
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_SIZE))

        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.rand(500)

        return embedding_matrix

    def get_embedded_input(self, review):
        tokenizer = self.get_tokenized()
        words = tokenizer.word_index
        embedding_matrix = self.get_embedding_matrix(tokenizer)

        json_train, json_test = self.__load_json()
        review = self.__read_data(json_train)
        review_test = self.__read_data(json_test)

        review_list = [review, review_test]

        for reviews in review_list:
            for review in reviews:
                splitted = review.split()
                temp = list()
                for word, i in splitted:
                    if word in words and i < MAX_LENGTH:
                        temp.append(embedding_matrix[words[word]])
                    else:
                        temp.append(np.zeros(EMBEDDING_SIZE))
                len_review = len(temp)
                for i in range(len_review, MAX_LENGTH):
                    temp.append(np.zeros(EMBEDDING_SIZE))
        return review

    def get_pos_matrix(self):
        return np.random.rand(27, 30)

    def get_encoded_pos(self, pos):
        return to_categorical(pos, num_classes = 26)