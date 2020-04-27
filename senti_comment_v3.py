import os
import matplotlib.pyplot as plt
import math
from os import listdir
# Vietnamese Tokenizer to segmentation
from pyvi import ViTokenizer
from collections import Counter
from gensim import corpora, matutils
from gensim import models
from sklearn import preprocessing
import numpy as np
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import re
import string
#import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import load_model
#import edit_post
import pymysql
from wordpress_xmlrpc import Client
from wordpress_xmlrpc.methods import media, posts
import csv

# load doc into memory
def load_doc(filename):
    #open the file as read only
    f = open(filename,"r",encoding="utf8", errors="ignore")
    #read all text
    text = f.read()
    #close the file
    f.close()
    return text
def read_stopword(filename):
    #open file
    f = open(filename,"r",encoding="utf16")
    stopword = set([word.strip().replace(' ','_') for word in f.readlines()])
    return stopword
def add_doc_to_vocab(filename, vocab):
    # load doc
    text = load_doc(filename)
    # clean doc
    tokens = clean_doc(text)
    # update vocab
    vocab.update(tokens)
    return vocab

# load all docs in a directory
def process_docs_build_dict(directory, vocab):
    #walk through all files in the folder
    for filename in listdir(directory):
        #skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next()
        #create the full path of the file to open
        path = directory + "/" + filename
        ##load document
        #doc = load_doc(path)
        # print('Load %s'%filename)
        add_doc_to_vocab(path,vocab)

def process_docs_filter_vocab(directory, vocab):
    #walk through all files in the folder
    lines = list()
    for filename in listdir(directory):
        #skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next()
        #create the full path of the file to open
        path = directory + "/" + filename
        ##load, clean, filter doc
        line = doc_to_line(path,vocab)
        # add to list
        lines.append(line)
    return lines

def count_character(directory):
    # walk through all files in the folder
    count_character_doc = list()
    word_doc = list()
    count_character_doc_clean = list()
    word_doc_clean = list()
    # i=0
    for filename in listdir(directory):
        # i=i+1
        # print('i: ',i)
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next()
        # create the full path of the file to open
        path = directory + "/" + filename
        #load, clean, filter doc
        doc = load_doc(path)
        doc_remove1 = doc.replace("_", " ")
        doc_remove2 = doc_remove1.replace("/", " / ")
        doc_remove3 = doc_remove2.replace("-", " - ")
        doc_split = doc_remove3.split()
        SPECIAL_CHARACTER = "0123456789?%@$.🏻\",<>=+-!;'/()*&^:#|[]{}~`"
        doc_special = [word.strip(SPECIAL_CHARACTER).lower() for word in doc_split]
        stopword = read_stopword(STOP_WORD_PATH)
        doc_clean_stopw = [word for word in doc_special if word not in stopword]
        while ("" in doc_clean_stopw):
            doc_clean_stopw.remove('')
        #word_doc.append(doc_split)
        for i in doc_split:
            count_char = len(i)
            # if(count_char > 10):
            #     print("filename: ", filename)
            #     print(i)
            count_character_doc.append(count_char)
            word_doc.append(i)

        for i in doc_clean_stopw:
            count_char = len(i)
            # if(count_char == 1):
            #     print("filename: ", filename)
            #     print("word-i:", i)
            count_character_doc_clean.append(count_char)
            word_doc_clean.append(i)

    #print("count_character_doc: ",count_character_doc)
    return count_character_doc, word_doc, count_character_doc_clean, word_doc_clean
def build_dic_word(word_doc_clean):
    vocab = Counter()
    #for i in word_doc_clean:
    vocab.update(word_doc_clean)
    min_occurence  = 1
    # update vocab
    tokens = [k for k,c in vocab.items() if c >= min_occurence]
    # dict_word = []
    # list_word = []
    # for k, c in vocab.items():
    #     if c>= min_occurence:
    #         list_word.append(k)
    # #print(list_word)
    # dict_word.append(list_word)
    # print(len(dict_word))

    #==== save tokens to dictionary =========#
    #=== using method 1: save dict as list of word
    save_list(tokens,DICTIONAY_PATH_WORD)
    print('finish generating the vocabratory of words that have not tokenized yet')

def process_docs_filter_vocab_dict(directory, vocab):
    #walk through all files in the folder
    lines = list()
    #i=0
    for filename in listdir(directory):
        #i=i+1
        #print('i: ',i)
        #skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next()
        #create the full path of the file to open
        path = directory + "/" + filename
        #print(path)
        ##load, clean, filter doc
        line = doc_to_line_dict(path,vocab)
        #print(line)
        # add to list
        lines.append(line)
    return lines

def clean_doc(doc):
    # 29.3.2020: comments are tokened, re-tokened using vntokenize
    doc_replace = doc.replace('_', ' ') #added for Tokenized doc.
    #tokens = ViTokenizer.tokenize(doc)
    tokens = ViTokenizer.tokenize(doc_replace)
    #tokens = ViTokenizer.tokenize(doc_replace)
    # split into tokens by white space
    token_split = tokens.split()
    #SPECIAL_CHARACTER = '0123456789?%@$.,<>=+-!;/()*"&^:#|[]{}~`\n\t\''
    SPECIAL_CHARACTER = '0123456789?%@$.,<>=+-!;/()*"&^:#|[]{}~`'
    # remove special character from each word
    # this differs English because "_" is used to tokenize
    tokens_special = [word.strip(SPECIAL_CHARACTER).lower() for word in token_split]
    #print('tokens_special:',tokens_special)
    # load stopword
    stopword = read_stopword(STOP_WORD_PATH)
    tokens_clean = [word for word in tokens_special if word not in stopword]
    #print('tokens_clean:',tokens_clean)
    #remove emty string "" in python
    while ("" in tokens_clean):
        tokens_clean.remove('')
    #print("tokens_clean removed "": ",tokens_clean)
    return tokens_clean
    #print(tokens)

def save_list(lines, filename):
    data = '\n'.join(lines)
    f = open(filename, 'w', encoding= "utf8")
    f.write(data)
    f.close()

def store_dictionary(filePath,dict_words):
    dictionary = corpora.Dictionary(dict_words)
    #dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n= 2000)
    #dictionary.filter_extremes(no_below=2, no_above=0.5)
    #dictionary.filter_extremes(no_below=20, no_above=0.3)
    dictionary.save_as_text(filePath)

def doc_to_line(filename, vocab):
    #load doc
    text = load_doc(filename)
    # clean doc
    tokens = clean_doc(text)
    # filter doc with vocab
    tokens = [w for w in tokens if w in vocab]
    #print(tokens)
    return ' '.join(tokens)

def doc_to_line_dict(filename, vocab):
    #load doc
    #list_word = []
    #print(filename)
    #print(vocab)
    #print(len(vocab))
    text = load_doc(filename)
    # clean doc
    tokens = clean_doc(text)
    #print(tokens)
    # filter doc with vocab
    tokens = [w for w in tokens if w in vocab]
    #print(tokens)
    #list_word.append(tokens)
    return tokens
    #return ' '.join(tokens)

def load_clean_dataset(vocab):
    # load docs
    neg = process_docs_filter_vocab(DATA_NEG_DIR_TRAIN, vocab)
    pos = process_docs_filter_vocab(DATA_POS_DIR_TRAIN, vocab)
    docs = neg+pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

def load_clean_dataset_dict(vocab):
    #load docs
    neg = process_docs_filter_vocab_dict(DATA_NEG_DIR_TRAIN, vocab)
    pos = process_docs_filter_vocab_dict(DATA_POS_DIR_TRAIN, vocab)
    #print(neg)
    docs = neg+pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

    # #=== only test pos that are error occurances
    # pos = process_docs_filter_vocab_dict(DATA_POS_DIR_TRAIN_TST, vocab)
    # # print(neg)
    # docs = pos
    # labels = [1 for _ in range(len(pos))]
    # return docs, labels

def load_clean_dataset_dict_path(vocab, path_neg, path_pos):
    # load docs
    neg = process_docs_filter_vocab_dict(path_neg, vocab)
    pos = process_docs_filter_vocab_dict(path_pos, vocab)
    docs = neg+pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def get_tfidf(documents):  # ??gensim????tfidf
    documents=[[word for word in document.text.split()] for document in documents]
    dictionary = corpora.Dictionary(documents)
    n_items = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in documents]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    ds = []
    for doc in corpus_tfidf:
        d = [0] * n_items
        for index, value in doc :
            d[index]  = value
        ds.append(d)
    return ds
def define_CNN_model(max_seq, embedding_size):
    model = Sequential()
    model.add(Conv1D(filters= 150, kernel_size= 3, activation= 'relu', input_shape=(max_seq,embedding_size)))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.1))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu'))
    model.add(Dense(1, activation= 'sigmoid'))
    #compile network
    model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    model.summary()
    return model

def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length= max_length))
    model.add(Conv1D(filters= 150, kernel_size= 3, activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.1))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu'))
    model.add(Dense(1, activation= 'sigmoid'))
    #compile network
    model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    model.summary()
    return model

def predict_comment_sentiment(review, word_labels,model_embedding, max_seq, embedding_size, model):
    # clean review
    doc_cleaned = clean_doc(review)
    #generate matrix for cleaned review
    review_matrix = comment_embedding(doc_cleaned, word_labels, model_embedding, max_seq, embedding_size)
    #print("review_matrix:",review_matrix)
    review_matrix_to_3D = review_matrix.reshape((1,max_seq,embedding_size))
    #print("review_matrix_to_3D.reshape: ",review_matrix_to_3D)

    ylabel = model.predict(review_matrix_to_3D, verbose = 0)
    percent_pos  = ylabel[0,0]
    if round(percent_pos) == 0:
        #return (1-percent_pos), 'NEGATIVE'
        return (1 - percent_pos), '0'
    #return percent_pos, 'POSITIVE'
    return percent_pos, '1'

def predict_sentiment(review, vocab, encoder, max_length, model):
    # clean review
    doc_cleaned = clean_doc(review)
    tokens = [w for w in doc_cleaned if w in vocab]
    encoded_doc = encoder.transform(tokens)
    padded_doc = pad_sequences([encoded_doc], maxlen= max_length)
    ylabel = model.predict(padded_doc, verbose = 0)
    percent_pos  = ylabel[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'
def comment_embedding(comment,word_labels, model_embedding, max_seq, embedding_size):
    matrix = np.zeros((max_seq, embedding_size))
    #words = comment.split()
    words = comment
    #print(words)
    lencmt = len(words)
    #print(lencmt)
    if(lencmt < max_seq):
        for i in range(lencmt):
            #print("i:", i)
            if(words[i] in word_labels):
                matrix[i] = model_embedding[words[i]]
                #print("matrix[i]:", matrix[i])
    else:
        for i in range(max_seq):
            #print("i:", i)
            if(words[i] in word_labels):
                matrix[i] = model_embedding[words[i]]
                #print("matrix[i]:", matrix[i])
    matrix = np.array(matrix)
    #print("matrix:",matrix)
    return matrix


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_NEG_DIR_TRAIN = os.path.join(DIR_PATH, 'sentiment_comments/data_train/data_train/train/neg')
DATA_POS_DIR_TRAIN = os.path.join(DIR_PATH, 'sentiment_comments/data_train/data_train/train/pos')
DATA_POS_DIR_TRAIN_TST = os.path.join(DIR_PATH, 'sentiment_comments/data_train_tst/data_train/train/pos')

DATA_NEG_DIR_VALID = os.path.join(DIR_PATH, 'sentiment_comments/data_train/data_train/test/neg')
DATA_POS_DIR_VALID = os.path.join(DIR_PATH, 'sentiment_comments/data_train/data_train/test/pos')

DATA_NEG_DIR_TEST = os.path.join(DIR_PATH, 'sentiment_comments/data_test/data_test/test/neg')
DATA_POS_DIR_TEST = os.path.join(DIR_PATH, 'sentiment_comments/data_test/data_test/test/pos')

STOP_WORD_PATH = os.path.join(DIR_PATH, 'stopwords-nlp-vi.txt')
DICTIONAY_PATH = os.path.join(DIR_PATH, 'dictionary_comment.txt')
DICTIONAY_PATH_WORD = os.path.join(DIR_PATH, 'dict_comment_word.txt')

#==== loading and pre-processing for data
# Xtrain_pre = load_clean_dataset_dict(vocab_load)
# len_xtrain = len(Xtrain)
# print("Xtrain length:",len(Xtrain))


# #====first process:
# # #====== step1: build dictionary using the training data ==============#
# # vocab = Counter()
# # process_docs_build_dict(DATA_NEG_DIR_TRAIN, vocab)
# # process_docs_build_dict(DATA_POS_DIR_TRAIN, vocab)
# # #print(vocab)
# # #print(vocab.items())
# # #print(vocab.keys())
# # # keep tokens with > 5 occurences
# # min_occurence  = 1
# # # update vocab
# # tokens = [k for k,c in vocab.items() if c >= min_occurence]
# # dict_word = []
# # list_word = []
# # for k, c in vocab.items():
# #     if c>= min_occurence:
# #         list_word.append(k)
# # #print(list_word)
# # dict_word.append(list_word)
# # print(len(dict_word))
# #
# # #==== save tokens to dictionary =========#
# # #=== using method 1: save dict as list of word
# # save_list(tokens,DICTIONAY_PATH)
# # print('finish generating the vocabotary')
# # #====== End build dictionary, this step is passed ==============#
#
# # step2: load dictionary
# vocab_load = load_doc(DICTIONAY_PATH)
# vocab_load = vocab_load.split()
# vocab_size = len(vocab_load)
# #vocab_load = set(vocab_load)
# #print(vocab_load)
# print("==== Length of dict: %d"%len(vocab_load))
# print(vocab_load)
#
# #==== step 2: start pre-processing dataset for training using the built vocab ===#
# #==== loading and pre-processing for data
# Xtrain, Ytrain = load_clean_dataset_dict(vocab_load)
# len_xtrain = len(Xtrain)
# print("Xtrain length:",len(Xtrain))
#
# # #===== step3: training word2vector model from Xtrain===========#
# # input_gensim = [ ]
# # for i in Xtrain:
# #     input_gensim.append(i)
# # #print(input_gensim)
# # print(len(input_gensim))
# # model = Word2Vec(input_gensim, size=100, window=3, min_count=0, workers=4, sg=1)
# # model.wv.save("word.model")
# ##=== This is passed after generating the word2vec model===#
#
#
# #==== step4: loading trained word2vector model ====#
# model_embedding = word2vec.KeyedVectors.load('./word.model')
# word_labels = []
# embedding_size = 100
# max_seq = 30
# for word in model_embedding.vocab.keys():
#     word_labels.append(word)
#     #print("word_labels",word_labels)
#
# # #=== convert comment to matrix with size of mxn: m is max_seq, n is word2vector size ====#
# # tst_str2= "Thích không_gian ở đây lắm luôn , rất đẹp , lại thấy gần_gũi , có chút cổ_điển xen_lẫn hiện_đại , lại có chỗ ngồi nhìn ra ngoài đường_ngắm cảnh , rất ok . Nước uống cũng không đến_nỗi tệ"
# # #tst_str = ViTokenizer.tokenize(tst_str)
# # tst_str_clean = clean_doc(tst_str2)
# # print(tst_str_clean)
# #
# # tst_matrix = comment_embedding(tst_str_clean,word_labels,model_embedding, max_seq, embedding_size)
# # print(tst_matrix)
# # print("tst_matrix[19]",tst_matrix[19])
# # print("tst_matrix[2]",tst_matrix[2])
# #===================================#
#
# #====== step5: generate 3D train_data by appending matrix cho each document ===#
# Xtrain_embedding = []
# for i in Xtrain:
#     tst_matrix = comment_embedding(i, word_labels, model_embedding, max_seq, embedding_size)
#     #print("tst_matrix.shape",tst_matrix.shape)
#     #print("tst_matrix:",tst_matrix)
#     tst_matrix = np.array(tst_matrix)
#     #tst_matrix_to_3D = tst_matrix.reshape((max_seq, embedding_size,1))
#     #print("Result: ", tst_matrix_to_3D)
#     Xtrain_embedding.append(tst_matrix)
#
# Xtrain_embedding = np.array(Xtrain_embedding).astype('float32')
# print("Xtrain_embedding.ndim: ",Xtrain_embedding.ndim)
# # print("Xtrain_embedding.shape: ", Xtrain_embedding.shape)
# # print("Xtrain_embedding: ",Xtrain_embedding)
# # print("Xtrain_embedding[0]:", Xtrain_embedding[0])
#
# #====== step6: prepare validation data ======#
# Xvalid, Yvalid = load_clean_dataset_dict_path(vocab_load, DATA_NEG_DIR_VALID, DATA_POS_DIR_VALID)
#
# #====== generate 3D train_data by appending matrix cho each document===#
# Xvalid_embedding = []
# for i in Xvalid:
#     tst_matrix = comment_embedding(i, word_labels, model_embedding, max_seq, embedding_size)
#     #print("tst_matrix.shape",tst_matrix.shape)
#     #print("tst_matrix:",tst_matrix)
#     tst_matrix = np.array(tst_matrix)
#     #tst_matrix_to_3D = tst_matrix.reshape((max_seq, embedding_size,1))
#     #print("Result: ", tst_matrix_to_3D)
#     Xvalid_embedding.append(tst_matrix)
#
# Xvalid_embedding = np.array(Xvalid_embedding).astype('float32')
# # print("Xvalid_embedding.shape: ", Xvalid_embedding.shape)
# # print("Xvalid_embedding: ",Xvalid_embedding)
# # print("Xvalid_embedding[0]:", Xvalid_embedding[0])
#
# #====== step 7: prepare data for testing======#
# Xtest, labels_test = load_clean_dataset_dict_path(vocab_load,DATA_NEG_DIR_TEST,DATA_POS_DIR_TEST)
# #====== generate 3D train_data by appending matrix cho each document===#
# Xtest_embedding = []
# for i in Xtest:
#     tst_matrix = comment_embedding(i, word_labels, model_embedding, max_seq, embedding_size)
#     #print("tst_matrix.shape",tst_matrix.shape)
#     #print("tst_matrix:",tst_matrix)
#     tst_matrix = np.array(tst_matrix)
#     #tst_matrix_to_3D = tst_matrix.reshape((max_seq, embedding_size,1))
#     #print("Result: ", tst_matrix_to_3D)
#     Xtest_embedding.append(tst_matrix)
#
# Xtest_embedding = np.array(Xtest_embedding).astype('float32')
# # print("Xtest_embedding.shape: ", Xtest_embedding.shape)
# # print("Xtest_embedding: ",Xtest_embedding)
# # print("Xtest_embedding[0]:", Xtest_embedding[0])
#
# # =======step 8: define model, train and save the model ===================#
# model = define_CNN_model(max_seq, embedding_size)
# # fit network
# model.fit(Xtrain_embedding, Ytrain, batch_size= 30, epochs= 50, verbose= 2, validation_data= (Xvalid_embedding, Yvalid))
# model.save('model_senti_comment_50itr.h5')
#
# #===== step 9: evaluate the model====#
# _, acc_model = model.evaluate(Xtest_embedding, labels_test, verbose= 0)
# print('Test Accuracy from current model: %f' % (acc_model*100))
#
# #======= step 10: loading the trained model ====#
# print('=== load model: ')
# #model_loaded = load_model('model_senti_1000v4.h5')
# model_loaded = load_model('model_senti_comment_50itr.h5')
#
# #====== step 11: test the real review/comment ===#
# text_tst = "mình gọi 1 phần cháo sườn gan 54k khác xa so với ảnh minh hoạ. cháo thì mặn chát không thể ăn được. \
# cảm thấy phí tiền @@"
#
# percentage, review_predict = predict_comment_sentiment(text_tst, word_labels,model_embedding, max_seq, embedding_size, model_loaded)
# print("review sentiment:", review_predict)
# print("percentage:", percentage)

if __name__ == '__main__':
    # count_character_doc, word_doc, count_character_doc_clean, word_doc_clean = count_character(DATA_POS_DIR_TRAIN_TST)
    # print("word_doc:",word_doc)
    # print(len(word_doc))
    # print(count_character_doc)
    # print(len(count_character_doc))
    #
    # print("word_doc_clean:",word_doc_clean)
    # print(len(word_doc_clean))
    # print(count_character_doc_clean)
    # print(len(count_character_doc_clean))
    #
    # max_leng = 0
    # int_index = 0
    # max_leng_clean = 0
    # int_index_clean = 0
    #
    # build_dic_word(word_doc_clean)
    #
    # plt.style.use('ggplot')
    # plt.hist(count_character_doc_clean, bins=range(10))
    # plt.show()

    #=== find the longest word and it's index in the list ===#
    # for i in count_character_doc:
    #     if i> max_leng:
    #         max_leng = i
    #         int_index = count_character_doc.index(i)
    # print(int_index)
    # print(word_doc[int_index])
    # print(max_leng)
    #======= implement sentiment analysis for text ==============#
    #==== step4: loading trained word2vector model ====#
    print("load model_embedding")
    model_embedding = word2vec.KeyedVectors.load('./word.model')
    word_labels = []
    embedding_size = 100
    max_seq = 30
    for word in model_embedding.vocab.keys():
        word_labels.append(word)
        #print("word_labels",word_labels)

    #======= step 10: loading the trained model ====#
    print('=== load model: ')
    #model_loaded = load_model('model_senti_1000v4.h5')
    model_loaded = load_model('model_senti_comment_50itr.h5')

    # #====== step 11: test the real review/comment ===#
    # text_tst = []
    # text_tst1 = "Nơi nào còn cộng sản là nơi đó còn đâu khổ"
    # text_tst2 = "Xhcn cái gì mà dân oan kêu khóc khắp nơi .oán hận ngút trời .bọn có chức có quyền thì cướp bóc vô độ."
    # text_tst3 = "Hay dung len va hay dung len Dong Bao oi"
    # text_tst4 = "Về nội dung: có nhiều bài phản giáo dục, bất lương. Việc bán sách, giá cả, số đầu sách không phù hợp mang tính độc quyền."
    # text_tst5 = "VN ĐÃ MẤT VÀO TÀU KHỰA RỒI❗MỞ MẮT RA MÀ RÁNG DÀNH LẠI NHE CU❗3/// CHỈ MONG KHÔNG MẤT TỔ QUỐC VN❗"
    # text_tst.append(text_tst1)
    # text_tst.append(text_tst2)
    # text_tst.append(text_tst3)
    # text_tst.append(text_tst4)
    # text_tst.append(text_tst5)
    # for i in text_tst:
    #     percentage, review_predict = predict_comment_sentiment(i, word_labels, model_embedding, max_seq,
    #                                                            embedding_size, model_loaded)
    #     print("Comment: %s \n ===> Sentiment: %s \n===> Probability: %f \n" % (i, review_predict, percentage))

    review_predict_list = []
    percentage_list = []
    import csv
    i=0
    with open("comment_senti_result.csv", "w+", encoding= 'utf-16') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'content', 'sentiment', 'probability'])
        with open('E:\Projects\\NLP\\sentiment\\comment_sentiment\\comments_3.csv',
                'r', encoding='utf-16') as file:
            #reader = csv.reader((line.replace('\0', '') for line in file))
            reader = csv.reader(file)
            for row in reader:
                print(row)
                #=== check and remove emotion ===#
                
                #out = []
                percentage, review_predict = predict_comment_sentiment(row[1],word_labels, model_embedding, max_seq, embedding_size, model_loaded)
                per = round(percentage,4)
                #add results to list
                review_predict_list.append(review_predict)
                percentage_list.append(percentage)
                #print(percentage_list)
                #print(review_predict_list)
                #print("Index: %s, content: %s \n ===> sentiment: %s, percentage: %f \n" % (row[0], row[1], review_predict, percentage))
                #out.append(row[0])
                #out.append(row[1])
                #out.append(review_predict)
                #out.append(percentage)
                writer.writerow([row[0],row[1],review_predict,per])
                print(i)
                i+=1
                #f.write(str(out)+"\n")


