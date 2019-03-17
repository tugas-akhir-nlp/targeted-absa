import json
import string
import numpy as np

from gensim.models import Word2Vec

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from DependencyTermExtractor import DependencyTermExtractor

MAX_LENGTH = 100
EMBEDDING_SIZE = 500
ASPECT_LIST = [
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
            masking=True,
            embedding=True,
            pos_tag='one_hot',
            dependency=True):
        self.normalize = normalize
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.masking = masking
        self.embedding = embedding
        self.pos_tag = pos_tag
        self.dependency = dependency

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
        return text

    def __mask_entity(self, text):
        return text

    def __load_embedding(self):
        print("Loading word embedding...")
        with open('resource/w2v_path.txt') as file:
            word2vec_path = file.readlines()[0]            
        w2v = Word2Vec.load(word2vec_path)
        return w2v

    def __load_json(self, path_file):
        with open(path_file) as f:
            json_data = json.load(f)
        return json_data

    def read_data_for_aspect(self, json_path):
        json_data = self.__load_json(json_path)
        review = list()
        for data in json_data:
            temp = self.__lower(data['text'])
            temp = self.__remove_punct(temp)
            review.append(temp)
        return review

    def __read_aspect(self, json_data):
        label = list()
        for i in range(len(json_data)):
            temp = np.zeros(len(ASPECT_LIST), dtype=int)
            for aspect in range(len(json_data[i]['aspects'])):
                if json_data[i]['aspects'] != []:
                    for j in range(len(ASPECT_LIST)):
                        if ASPECT_LIST[j] in json_data[i]['aspects'][aspect]:
                            temp[j] = 1
            label.append(temp)
        return np.array(label)

    def read_data_for_sentiment(self, json_path):
        json_data = self.__load_json(json_path)
        review = list()
        for data in json_data:
            data_unique = np.unique(data['aspect'], axis=0)
            for i in range(len(data_unique)):
                temp = self.__lower(data['text'])
                temp = self.__remove_punct(temp)
                review.append(temp)
        return review

    def __read_sentiment(self, json_data):
        label = np.zeros(len(json_data), dtype=int)
        idx = 0
        for data in json_data:
            data_unique = np.unique(data['aspect'], axis=0)
            for i in range(len(data_unique)):
                if 'positive' in data_unique[i][1]:
                    label[idx] = 1
                idx += 1
        label = to_categorical(label, num_classes=2)
        return np.array(label)
    
    def __read_pos(self, json_data):
        pos = list()
        pos_dict = self.__load_json('resource/pos_dict.json')
        
        for data in json_data:
            temp = np.zeros(MAX_LENGTH, dtype=int)
            idx = 0
            for sentence in data['sentences']:
                for i, token in enumerate(sentence['tokens']):
                    if i >= MAX_LENGTH:
                        continue
                    temp[idx] = pos_dict[token['pos_tag']]
                    idx += 1
            pos.append(temp)
        return pos

    def get_tokenized(self):
        review = self.read_data_for_aspect('aspect/data/aspect_train.json')

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(review)
        return tokenizer

    def get_vocab_size(self, tokenizer):
        return len(tokenizer.word_index) + 1

    def get_encoded_input(self):
        tokenizer = self.get_tokenized()

        review = self.read_data_for_aspect('aspect/data/aspect_train.json')
        review_test = self.read_data_for_aspect('aspect/data/aspect_test.json')

        encoded_data = tokenizer.texts_to_sequences(review)
        encoded_data_test = tokenizer.texts_to_sequences(review_test)

        x_train = pad_sequences(encoded_data, maxlen=MAX_LENGTH, padding='post')
        x_test = pad_sequences(encoded_data_test, maxlen=MAX_LENGTH, padding='post')

        return x_train, x_test

    def get_embedding_matrix(self, tokenizer):
        w2v = self.__load_embedding()
        words = list(w2v.wv.vocab)
        embeddings_index = dict()

        print("Creating embedding matrix...")

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

    def get_embedded_input(self):
        w2v = self.__load_embedding()

        words = list(w2v.wv.vocab)

        print("Getting all embedding vector...")

        review = self.read_data_for_aspect('aspect/data/aspect_train.json')
        review_test = self.read_data_for_aspect('aspect/data/aspect_test.json')

        review_list = [review, review_test]

        x_train = list()
        x_test = list()

        x_list = [x_train, x_test]

        for i, reviews in enumerate(review_list):
            for review in reviews:
                splitted = review.split()
                temp = list()
                for j, word in enumerate(splitted):
                    if word in words and i < MAX_LENGTH:
                        temp.append(w2v[word])
                    else:
                        temp.append(np.zeros(EMBEDDING_SIZE))
                len_review = len(temp)
                for j in range(len_review, MAX_LENGTH):
                    temp.append(np.zeros(EMBEDDING_SIZE))
                x_list[i].append(temp)
            
        return x_train, x_test

    def get_pos_matrix(self):
        return np.random.rand(27, 30)

    def get_encoded_pos(self, pos):
        return to_categorical(pos, num_classes = 26)

    def get_encoded_term(self, trees):
        e = DependencyTermExtractor()
        term = list()
        for i, tree in enumerate(trees):
            temp = e.get_position_target(tree)
            term.append(temp)
        term = to_categorical(term, num_classes=2)
        return term

    def concatenate(self, sentences_a, sentences_b):
        concat = list()
        for i, sentence in enumerate(sentences_a):
            temp = list()
            for j, word in enumerate(sentence):
                temp.append(np.concatenate((word, sentences_b[i][j]), axis=0))
            concat.append(temp)
        return np.array(concat)

    def get_all_input(self):
        if self.embedding:
            x_train, x_test = self.get_encoded_input()
        else:
            x_train, x_test = self.get_embedded_input()
            if self.pos_tag is 'one_hot':
                json_train = self.__load_json('resource/postag_train_input.json')
                json_test = self.__load_json('resource/postag_test_input.json')

                pos_train = self.__read_pos(json_train)
                pos_test = self.__read_pos(json_test)

                encoded_train = self.get_encoded_pos(pos_train)
                encoded_test = self.get_encoded_pos(pos_test)

                x_train = self.concatenate(x_train, encoded_train)
                x_test = self.concatenate(x_test, encoded_test)

            if self.dependency is True:
                json_train = self.__load_json('resource/dependency_train_auto.json')
                json_test = self.__load_json('resource/dependency_train_auto.json')

                encoded_train = self.get_encoded_term(json_train)
                encoded_test = self.get_encoded_term(json_test)

                x_train = self.concatenate(x_train, encoded_train)
                x_test = self.concatenate(x_test, encoded_test)

        json_train = self.__load_json('aspect/data/aspect_train.json')
        json_test = self.__load_json('aspect/data/aspect_test.json')
        y_train = self.__read_aspect(json_train)
        y_test = self.__read_aspect(json_test)

        return x_train, y_train, x_test, y_test