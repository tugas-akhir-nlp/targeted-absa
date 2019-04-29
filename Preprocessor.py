import re
import json
import string
import numpy as np

from gensim.models import Word2Vec

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from DependencyTermExtractor import DependencyTermExtractor

MAX_LENGTH = 50
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
            module_name = 'aspect',
            train_file = None,
            test_file = None,
            lowercase = True,
            remove_punct = True,
            embedding = True,
            pos_tag = 'embedding',
            dependency = True,
            use_entity = True,
            position_embd = False,
            mask_entity = False, 
            use_lexicon = None,
            use_op_target = None):
        self.module_name = module_name
        self.train_file = train_file
        self.test_file = test_file
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.embedding = embedding
        self.pos_tag = pos_tag
        self.dependency = dependency
        self.use_entity = use_entity
        self.position_embd = position_embd
        self.mask_entity = mask_entity
        self.use_lexicon = use_lexicon
        self.use_op_target = use_op_target

        self.punctuations = set(string.punctuation)

    def __lower(self, text):
        return text.lower() if self.lowercase else text

    def __remove_punct(self, text):
        if self.remove_punct:
            if self.mask_entity:
                clean = re.sub(r"[,.;@?!&$]+\ *", " ", text)
            else:
                clean = re.sub(r"[,.;@#?!&$]+\ *", " ", text)
        return clean

    def __load_embedding(self):
        print("Loading word embedding...")
        with open('resource/w2v_path.txt') as file:
            word2vec_path = file.readlines()[0]            
        w2v = Word2Vec.load(word2vec_path)
        return w2v

    def __load_json(self, path_file):
        with open(path_file, 'r') as f:
            json_data = json.load(f)
        return json_data

    def __load_txt(self, path_file):
        with open(path_file, 'r', encoding='utf-8') as f:
            txt_data = f.read().splitlines()
        return txt_data

    def read_data_for_aspect(self, json_path):
        json_data = self.__load_json(json_path)
        review = list()
        if self.use_entity:
            for data in json_data:
                for _ in data['info']:
                    if self.mask_entity:
                        temp = self.__lower(data['masked_sentence'])
                    else:
                        temp = self.__lower(data['sentence'])
                    temp = self.__remove_punct(temp)
                    review.append(temp)
        else:
            for data in json_data:
                if self.mask_entity:
                    temp = self.__lower(data['masked_sentence'])
                else:
                    temp = self.__lower(data['sentence'])
                temp = self.__remove_punct(temp)
                review.append(temp)
        return review

    def __read_aspect(self, json_path):
        label = list()
        data = self.__load_json(json_path)

        if self.use_entity:
            for i, datum in enumerate(data):
                for info in datum['info']:
                    temp = list()
                    for aspect in info['aspect']:
                        temp.append(aspect.split('|')[0])
                    label.append(temp)
        else:
            for i, datum in enumerate(data):
                temp = list()
                for info in datum['info']:
                    for aspect in info['aspect']:
                        temp.append(aspect.split('|')[0])
                label.append(temp)
        
        encoded_label = list()
        for aspects in label:
            temp = np.zeros(len(ASPECT_LIST), dtype=int)
            for aspect in aspects:
                for i, asp in enumerate(ASPECT_LIST):
                    if asp in aspect:
                        temp[i] = 1
            encoded_label.append(temp)

        print('Label shape  :', np.array(encoded_label).shape)    
        print('Example label:', encoded_label[0])
            
        return np.array(encoded_label)

    def read_data_for_sentiment(self, json_path):
        data = self.__load_json(json_path)
        review = list()
        if self.use_entity:
            for datum in data:
                for info in datum['info']:
                    for aspect in info['aspect']:
                        if self.mask_entity:
                            temp = self.__lower(datum['masked_sentence'])
                        else:
                            temp = self.__lower(datum['sentence'])
                        temp = self.__remove_punct(temp)
                        review.append(temp)
        # else:
        #     for datum in data:
        #         for info in datum['info']:
        #             temp = self.__lower(datum['sentence'])
        #             temp = self.__remove_punct(temp)
        #             review.append(temp)
        return review
    
    def __aspect2idx(self, data):
        print('panjang data: ', len(data))
        new = np.zeros(len(data))
        for i in range(len(data)):
            if data[i] == 'others':
                new[i] = 0
            elif data[i] == 'machine':
                new[i] = 1
            elif data[i] == 'part':
                new[i] = 2
            elif data[i] == 'price':
                new[i] = 3
            elif data[i] == 'service':
                new[i] = 4
            elif data[i] == 'fuel':
                new[i] = 5
        if not self.embedding:
            new = to_categorical(new, num_classes=6)
        return new

    def read_sentiment(self, json_path, review):
        data = self.__load_json(json_path)
        label = list()
        aspects = list()

        if self.use_entity:
            for i, datum in enumerate(data):        
                for info in datum['info']:
                    for aspect in info['aspect']:
                        if aspect.split('|')[1] == 'negative':
                            label.append(0)
                        elif aspect.split('|')[1] == 'positive':
                            label.append(1)
                        aspects.append(aspect.split('|')[0])
            # for i in range(len(review)):
            #     print(i)
            #     print(review[i])
            #     print(aspects[i], label[i])
            label = to_categorical(label, num_classes=2)

            aspects = self.__aspect2idx(aspects)
            new_aspect = list()
            for asp in aspects:
                temp = list()
                for i in range(MAX_LENGTH):
                    temp.append(asp)
                new_aspect.append(temp)
            print('Sentiment aspect shape :', np.array(new_aspect).shape)
        
        return np.array(label), np.array(new_aspect)

    def get_entities(self, json_path):
        data = self.__load_json(json_path)
        entities = list()

        for datum in data:
            for info in datum['info']:
                # if self.module_name == 'aspect':
                if info['name'] != None:
                    entities.append(info['name'])
                else:
                    entities.append('None')
                # elif self.module_name == 'sentiment':
                #     if info['name'] != None:
                #         for aspect in info['aspect']:
                #             entities.append(info['name'])
                #     else:
                #         for aspect in info['aspect']:
                #             entities.append('None')
        return entities

    def get_pos_dict(self):
        pos_dict = self.__load_json('resource/pos_dict.json')
        pos_size = len(pos_dict) + 2
        return pos_dict, pos_size
    
    def read_pos(self, json_path):
        pos = list()
        pos_dict, _ = self.get_pos_dict()
        pos_data = self.__load_json(json_path)
        
        if json_path == 'resource/postag_train_auto.json':
            json_data = self.__load_json(self.train_file)
        elif json_path == 'resource/postag_test_auto.json':
            json_data = self.__load_json(self.test_file)

        for i, data in enumerate(json_data):
            temp = np.zeros(MAX_LENGTH, dtype=int)
            idx = 0
            for j in range(len(pos_data[i]['sentences'])):
                for token in pos_data[i]['sentences'][j]['tokens']:
                    if self.remove_punct:
                        if pos_dict[token['pos_tag']] != 'PUN':
                            temp[idx] = pos_dict[token['pos_tag']]
                            idx += 1
                    else:
                        temp[idx] = pos_dict[token['pos_tag']]
                        idx += 1
                    if idx == MAX_LENGTH - 1:
                        break  
                if idx == MAX_LENGTH - 1:
                        break  
            if self.module_name == 'aspect':
                if self.use_entity:
                    for _ in data['info']:
                        pos.append(temp)
                else:
                    pos.append(temp)
            elif self.module_name == 'sentiment':
                if self.use_entity:
                    for info in data['info']:
                        for _ in info['aspect']:
                            pos.append(temp)

        pos = np.array(pos)
        return pos
    
    def get_sentiment_lexicons(self, file):
        with open('resource/positif.txt', 'r', encoding='utf-8') as f:
            positive = f.read().splitlines()
        with open('resource/negatif.txt', 'r', encoding='utf-8') as f:
            negative = f.read().splitlines()

        pos_neg_all = list()
        review = self.read_data_for_sentiment(file)
        for sentence in review:
            pos_neg = np.zeros(MAX_LENGTH)
            for neg in negative:
                if neg in sentence:
                    split_neg = neg.split()
                    len_neg = len(split_neg)
                    split_sen = sentence.split()
                    for i, token in enumerate(split_sen):
                        if split_neg[0] in token:
                            for j in range(i, i+len_neg):
                                pos_neg[j] = 2
                                if j == MAX_LENGTH - 1:
                                    break
                        if i == MAX_LENGTH - 1:
                            break
                        
            for pos in positive:
                if pos in sentence:
                    split_pos = pos.split()
                    len_pos = len(split_pos)
                    split_sen = sentence.split()
                    for i, token in enumerate(split_sen):
                        if split_pos[0] in token:
                            for j in range(i, i+len_pos):
                                if pos_neg[j] != 2:
                                    pos_neg[j] = 1
                                if j == MAX_LENGTH - 1:
                                    break
                        if i == MAX_LENGTH - 1:
                            break
            pos_neg_all.append(pos_neg)
        return np.array(pos_neg_all)

    def get_positional_embedding_without_masking(self, entity_path):
        list_position = list()
        entity_json = self.__load_json(entity_path)
        for sentence in entity_json:            
            for ent in sentence['info']:
                if ent['name'] != None:                
                    dist = 0
                    position = list()
                    entity = ent['name'].lower()
                    entity = re.sub('ku', '', entity)
                    entity = re.sub('-nya', '', entity)
                    entity = re.sub('nya', '', entity)
                    e_split = entity.split()
                    e_first = e_split[0]
                    split = (sentence['sentence'].lower()).split()

                    for token in split:
                        if e_first in token:
                            loc = split.index(token)
                            break
                    dist = dist - loc
                    for j in range(0,loc):
                        position.append(dist)
                        dist += 1
                    for j in range(loc,loc+len(e_split)):
                        position.append(dist)
                    for j in range(loc+len(e_split), len(split)):
                        dist += 1
                        position.append(dist)
                    for j in range(len(split), MAX_LENGTH):
                        position.append(1000)
                    position = position[:MAX_LENGTH]
                    if self.module_name == 'aspect':
                        # print(split)
                        # print(position)
                        # print('=======================================================')
                        list_position.append(position)
                    elif self.module_name == 'sentiment':
                        for _ in ent['aspect']:
                            # print(split)
                            # print(position)
                            # print('=======================================================')
                            list_position.append(position)
                else:
                    dist = 0
                    position = list()
                    split = (sentence['sentence'].lower()).split()

                    for j in range(0, len(split)):
                        position.append(dist)
                    for j in range(len(split), MAX_LENGTH):
                        position.append(1000)
                    position = position[:MAX_LENGTH]
                    if self.module_name == 'aspect':
                        # print(split)
                        # print(position)
                        # print('=======================================================')
                        list_position.append(position)
                    elif self.module_name == 'sentiment':
                        for _ in ent['aspect']:
                            # print(split)
                            # print(position)
                            # print('=======================================================')
                            list_position.append(position)
            
        return np.array(list_position)


    def get_positional_embedding_with_masking(self, entity_path):
        position = list()
        entity_file = self.__load_json(entity_path)
        for a, review in enumerate(entity_file):
            split = (review['masked_sentence'].lower()).split()   
            for name in review['info']:
                if name['name'] != None:
                    ent_name = name['entity_name']
                    for i, token in enumerate(split):
                        if ent_name in token:
                            temp = list()
                            dist = 0 - i
                            for j in range(0, i):
                                temp.append(dist)
                                dist += 1
                            dist = 0
                            temp.append(dist)
                            for j in range(i+1, len(split)):
                                dist += 1
                                temp.append(dist)
                            for j in range(len(split), MAX_LENGTH):
                                temp.append(1000)
                            if self.module_name == 'aspect':
                                position.append(temp[:MAX_LENGTH])
                            elif self.module_name == 'sentiment':
                                for _ in name['aspect']:
                                    position.append(temp[:MAX_LENGTH])
                            
                else:
                    temp = list()
                    for j in range(0, len(split)):
                        temp.append(0)
                    for j in range(len(split), MAX_LENGTH):
                        temp.append(1000)
                    if self.module_name == 'aspect':
                        position.append(temp[:MAX_LENGTH])
                    elif self.module_name == 'sentiment':
                        for _ in name['aspect']:
                            position.append(temp[:MAX_LENGTH])
        return np.array(position)


    def get_tokenized(self):
        review = self.read_data_for_aspect(self.train_file)

        if self.remove_punct:
            tokenizer = Tokenizer(oov_token=True)
        else:
            tokenizer = Tokenizer(filters='', oov_token=True)
        tokenizer.fit_on_texts(review)
        return tokenizer

    def get_vocab_size(self, tokenizer):
        return len(tokenizer.word_index) + 1

    def get_encoded_input(self):
        tokenizer = self.get_tokenized()

        if self.module_name == 'aspect':
            review = self.read_data_for_aspect(self.train_file)
            review_test = self.read_data_for_aspect(self.test_file)
        elif self.module_name == 'sentiment':
            review = self.read_data_for_sentiment(self.train_file)
            review_test = self.read_data_for_sentiment(self.test_file)

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

        if self.module_name is 'aspect':
            review = self.read_data_for_aspect(self.train_file)
            review_test = self.read_data_for_aspect(self.test_file)
            print("Successfully read aspect data")
        elif self.module_name is 'sentiment':
            review = self.read_data_for_sentiment(self.train_file)
            review_test = self.read_data_for_sentiment(self.test_file)
            print("Successfully read sentiment data")

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
        return np.array(x_train), np.array(x_test)

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

    def __concatenate(self, sentences_a, sentences_b):
        concat = list()
        print('dimensi word:', sentences_a.shape)
        print('dimensi aspek:', sentences_b.shape)
        for i, sentence in enumerate(sentences_a):
            temp = list()
            for j, word in enumerate(sentence):
                temp.append(np.concatenate((word, sentences_b[i][j]), axis=0))
            concat.append(temp)
        return np.array(concat)

    def get_all_input_aspect(self):
        if self.embedding:
            x_train, x_test = self.get_encoded_input()
        else:
            x_train, x_test = self.get_embedded_input()
            if self.pos_tag == 'one_hot':
                pos_train = self.read_pos('resource/postag_train_auto.json')
                pos_test = self.read_pos('resource/postag_test_auto.json')

                encoded_train = self.get_encoded_pos(pos_train)
                encoded_test = self.get_encoded_pos(pos_test)

                x_train = self.__concatenate(x_train, encoded_train)
                x_test = self.__concatenate(x_test, encoded_test)

            if self.dependency:
                json_train = self.__load_json('resource/dependency_train_auto.json')
                json_test = self.__load_json('resource/dependency_train_auto.json')

                encoded_train = self.get_encoded_term(json_train)
                encoded_test = self.get_encoded_term(json_test)

                x_train = self.__concatenate(x_train, encoded_train)
                x_test = self.__concatenate(x_test, encoded_test)
             
        y_train = self.__read_aspect(self.train_file)
        y_test = self.__read_aspect(self.test_file)

        return x_train, y_train, x_test, y_test

    def get_all_input_sentiment(self):
        if self.embedding:
            x_train, x_test = self.get_encoded_input()
            y_train, aspect_train = self.read_sentiment(self.train_file, x_train)
            y_test, aspect_test = self.read_sentiment(self.test_file, x_test)
        
        else:
            x_train, x_test = self.get_embedded_input()
            y_train, aspect_train = self.read_sentiment(self.train_file, x_train)
            y_test, aspect_test = self.read_sentiment(self.test_file, x_test)

            x_train = self.__concatenate(x_train, aspect_train)
            x_test = self.__concatenate(x_test, aspect_test)

            if self.pos_tag == 'one_hot':
                pos_train = self.read_pos('resource/postag_train_auto.json')
                pos_test = self.read_pos('resource/postag_test_auto.json')

                encoded_train = self.get_encoded_pos(pos_train)
                encoded_test = self.get_encoded_pos(pos_test)

                x_train = self.__concatenate(x_train, encoded_train)
                x_test = self.__concatenate(x_test, encoded_test)

            if self.dependency:
                json_train = self.__load_json('resource/dependency_train_auto.json')
                json_test = self.__load_json('resource/dependency_train_auto.json')

                encoded_train = self.get_encoded_term(json_train)
                encoded_test = self.get_encoded_term(json_test)

                x_train = self.__concatenate(x_train, encoded_train)
                x_test = self.__concatenate(x_test, encoded_test)      

        return x_train, y_train, x_test, y_test