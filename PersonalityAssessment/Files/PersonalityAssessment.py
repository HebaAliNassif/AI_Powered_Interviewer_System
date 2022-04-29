import numpy as np
import pandas as pd

import nltk 
import num2words 
import re
import pickle
import os
import time
import string

from sklearn.model_selection import train_test_split
from nltk import wordpunct_tokenize, word_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']


def loadDataset(): 
    # Load from .csv file with complete dataset
    data_essays = pd.read_csv('dataset/essays.csv', encoding = "ISO-8859-1")
    data_essays['cEXT'] = np.where(data_essays['cEXT']=='y', 1, 0)
    data_essays['cNEU'] = np.where(data_essays['cNEU']=='y', 1, 0)
    data_essays['cAGR'] = np.where(data_essays['cAGR']=='y', 1, 0)
    data_essays['cCON'] = np.where(data_essays['cCON']=='y', 1, 0)
    data_essays['cOPN'] = np.where(data_essays['cOPN']=='y', 1, 0)

    X_essays = data_essays['TEXT']
    y_essays = data_essays[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

    #data_essays['text length'] = data_essays['TEXT'].apply(len)

    labels = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    X_train, X_test, y_train, y_test = train_test_split(X_essays, y_essays, test_size=0.2)
    return X_train, X_test, y_train, y_test


def clean_text(data):
    data = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", data)
    data = re.sub(r"what's", "what is ", data)
    data = re.sub(r"\'s", " ", data)
    data = re.sub(r"\'ve", " have ", data)
    data = re.sub(r"can't", "cannot ", data)
    data = re.sub(r"n't", " not ", data)
    data = re.sub(r"I'm", "I am ", data)
    data = re.sub(r"\'re", " are ", data)
    data = re.sub(r"\'d", " would ", data)
    data = re.sub(r"\'ll", " will ", data)
    
    data = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), data)

    #data = re.sub(r'[^\w\s]', '', data)
    #data = re.sub(r"(\d+)(k)", r"\g<1>000", data)
    return data    



def lemmatize_token(token, tag):
    tag = {
        'N': nltk.corpus.wordnet.NOUN,
        'V': nltk.corpus.wordnet.VERB,
        'R': nltk.corpus.wordnet.ADV,
        'J': nltk.corpus.wordnet.ADJ
    }.get(tag[0], nltk.corpus.wordnet.NOUN)
    return WordNetLemmatizer().lemmatize(token, tag)


def preprocess_text(X):
    """
    Returns a preprocessed version of a full corpus (ie. tokenization and lemmatization using POS taggs)
    """
    #X = ' '.join(X_corpus)
    lemmatized_tokens = []


    # Clean the text
    X = clean_text(X)


    # Break the text into sentences
    for sent in sent_tokenize(X):
        
        # Remove punctuation
        #sent = remove_punctuation(sent)

        # Break the sentence into part of speech tagged tokens
        for token, tag in pos_tag(word_tokenize(sent)):

            # Apply preprocessing to the token
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If punctuation or stopword, ignore token and continue
            #if token in set(stopwords.words('english')) or all(char in set(string.punctuation) for char in token):

            # Removing stopwords, punctuation and one-letter words
            if token in set(stop_words) or token in punctuations or len(token) == 1:
                continue

            # Lemmatize the token
            lemma = lemmatize_token(token, tag)
            lemmatized_tokens.append(lemma)


    doc = ' '.join(lemmatized_tokens)

    return doc    

def preprocessDataset(data):
    numberOfDocuments = len(data)
    dataset = []
    for i in range(numberOfDocuments):
        dataset.append(preprocess_text(data.iloc[i]))
    return dataset


def calculateDF(data):
    numberOfDocuments = len(data)
    DF = {}
    #Set word as the key and the list of doc id’s as the value
    for i in range(numberOfDocuments):
        tokens = data[i]

        for w in tokens.split():
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

    # Replace the list of docs with its count
    for i in DF:
        DF[i] = len(DF[i])      

    return DF      


def doc_freq(text, s): 
    if text in s:
        loc = list(s).index(text)
        value_at_index = list(s.values())[loc]
        return value_at_index


def calculateTFIDF(data, DF):
    tf_idf = {}
    N = len(data)

    for i in range(N):
        tokens = data[i].split()
        words_count = len(tokens)
        counter = Counter(tokens)
        for token in np.unique(tokens):
            tf = counter[token]/words_count
            df = doc_freq(token, DF)
            idf = np.log(N/(df+1))
            tf_idf[i, token] = tf*idf

    return tf_idf

def trainingDataVectorization(dataset, tf_idf, vocab):
    vectors = np.zeros((len(dataset), len(vocab)))
    for i in tf_idf:
        ind = vocab.index(i[1])
        vectors[i[0]][ind] = tf_idf[i]

    return vectors

def testingDataVectorization(dataset, tf_idf, vocab):
    vectors = np.zeros((len(dataset), len(vocab)))
    for i in tf_idf:
        if(i[1] in vocab):
            ind = vocab.index(i[1])
            vectors[i[0]][ind] = tf_idf[i]

    return vectors

def prepareModel(y_train, y_test):
    train_y_cEXT = y_train.iloc[:, 0]
    train_y_cNEU = y_train.iloc[:, 1]
    train_y_cAGR = y_train.iloc[:, 2]
    train_y_cCON = y_train.iloc[:, 3]
    train_y_cOPN = y_train.iloc[:, 4]

    test_y_cEXT = y_test.iloc[:, 0]
    test_y_cNEU = y_test.iloc[:, 1]
    test_y_cAGR = y_test.iloc[:, 2]
    test_y_cCON = y_test.iloc[:, 3]
    test_y_cOPN = y_test.iloc[:, 4]

    return train_y_cEXT, train_y_cNEU, train_y_cAGR, train_y_cCON, train_y_cOPN, test_y_cEXT, test_y_cNEU, test_y_cAGR, test_y_cCON, test_y_cOPN


def trainModel(train_x_vectors, train_y_cEXT, train_y_cNEU, train_y_cAGR, train_y_cCON, train_y_cOPN):
    name="RF"

    print("training Extraversion cEXT using Random Forest...")
    clf_rf_cEXT = RandomForestClassifier(n_estimators=100)
    clf_rf_cEXT.fit(train_x_vectors, train_y_cEXT)
    print("cEXT score: ", clf_rf_cEXT.score(test_x_vectors, test_y_cEXT))

    print("training Neuroticism cNEU using Random Forest...")
    clf_rf_cNEU = RandomForestClassifier(n_estimators=100)
    clf_rf_cNEU.fit(train_x_vectors, train_y_cNEU)
    print("cNEU score: ", clf_rf_cNEU.score(test_x_vectors, test_y_cNEU))

    print("training Agreeableness cAGR using using Random Forest...")
    clf_rf_cAGR = RandomForestClassifier(n_estimators=100)
    clf_rf_cAGR.fit(train_x_vectors, train_y_cAGR)
    print("cAGR score: ", clf_rf_cAGR.score(test_x_vectors, test_y_cAGR))

    print("training Conscientiousness cCON using Random Forest...")
    clf_rf_cCON = RandomForestClassifier(n_estimators=100)
    clf_rf_cCON.fit(train_x_vectors, train_y_cCON)
    print("cCON score: ", clf_rf_cCON.score(test_x_vectors, test_y_cCON))
 
    print("training Openness to Experience cOPN using Random Forest...")
    clf_rf_cOPN = RandomForestClassifier(n_estimators=100)
    clf_rf_cOPN.fit(train_x_vectors, train_y_cOPN)
    print("cOPN score: ", clf_rf_cOPN.score(test_x_vectors, test_y_cOPN))

    return clf_rf_cEXT, clf_rf_cNEU, clf_rf_cAGR, clf_rf_cCON, clf_rf_cOPN

def saveModel(clf_rf_cEXT, clf_rf_cNEU, clf_rf_cAGR, clf_rf_cCON, clf_rf_cOPN):  
    RF_cEXT_model = 'models/RF_cEXT_model.sav'
    pickle.dump(clf_rf_cEXT, open(RF_cEXT_model, 'wb'))

    RF_cNEU_model = 'models/RF_cNEU_model.sav'
    pickle.dump(clf_rf_cNEU, open(RF_cNEU_model, 'wb'))

    RF_cAGR_model = 'models/RF_cAGR_model.sav'
    pickle.dump(clf_rf_cAGR, open(RF_cAGR_model, 'wb'))

    RF_cCON_model = 'models/RF_cEXT_model.sav'
    pickle.dump(clf_rf_cCON, open(RF_cCON_model, 'wb'))

    RF_cOPN_model = 'models/RF_cOPN_model.sav'
    pickle.dump(clf_rf_cOPN, open(RF_cOPN_model, 'wb'))

def loadModel():
    RF_cEXT_model = 'models/RF_cEXT_model.sav'
    clf_rf_cEXT = pickle.load(open(RF_cEXT_model, 'rb'))

    RF_cNEU_model = 'models/RF_cNEU_model.sav'
    clf_rf_cNEU = pickle.load(open(RF_cNEU_model, 'rb'))
    
    RF_cAGR_model = 'models/RF_cAGR_model.sav'
    clf_rf_cAGR = pickle.load(open(RF_cAGR_model, 'rb'))

    RF_cCON_model = 'models/RF_cEXT_model.sav'
    clf_rf_cCON = pickle.load(open(RF_cCON_model, 'rb'))

    RF_cOPN_model = 'models/RF_cOPN_model.sav'
    clf_rf_cOPN = pickle.load(open(RF_cOPN_model, 'rb'))

    return clf_rf_cEXT, clf_rf_cNEU, clf_rf_cAGR, clf_rf_cCON, clf_rf_cOPN
    

def predictModel(Answers_vectors):
    clf_rf_cEXT, clf_rf_cNEU, clf_rf_cAGR, clf_rf_cCON, clf_rf_cOPN = loadModel()

    cEXT = clf_rf_cEXT.predict_proba(Answers_vectors)
    cEXT = np.array(cEXT)
    cEXTAverage = (cEXT.sum(axis = 0)[1]*100)/len(cEXT)

    cNEU = clf_rf_cNEU.predict_proba(Answers_vectors)
    cNEU = np.array(cNEU)
    cNEUAverage = (cNEU.sum(axis = 0)[1]*100)/len(cNEU)

    cAGR = clf_rf_cAGR.predict_proba(Answers_vectors)
    cAGR = np.array(cAGR)
    cAGRAverage = (cAGR.sum(axis = 0)[1]*100)/len(cAGR)

    cCON = clf_rf_cCON.predict_proba(Answers_vectors)
    cCON = np.array(cCON)
    cCONAverage = (cCON.sum(axis = 0)[1]*100)/len(cCON)

    cOPN = clf_rf_cOPN.predict_proba(Answers_vectors)
    cOPN = np.array(cOPN)
    cOPNAverage = (cOPN.sum(axis = 0)[1]*100)/len(cOPN)

    return cEXTAverage, cNEUAverage, cAGRAverage, cCONAverage, cOPNAverage

def predictPersonality(Answers, vocab):
    Answers_dataset = Answers

    numberOfAnswers = len(Answers)

    Answers_DF = {}

    #Set word as the key and the list of doc id’s as the value
    for i in range(numberOfAnswers):
        Answers_tokens = Answers_dataset[i]

        for w in Answers_tokens.split():
            try:
                Answers_DF[w].add(i)
            except:
                Answers_DF[w] = {i}

    # Replace the list of docs with its count
    for i in Answers_DF:
        Answers_DF[i] = len(Answers_DF[i])            

    Answers_tf_idf = {}
    Answers_N = len(Answers_dataset)

    for i in range(Answers_N):
        Answers_tokens = Answers_dataset[i].split()
        Answers_words_count = len(Answers_tokens)
        Answers_counter = Counter(Answers_tokens)
        for Answers_token in np.unique(Answers_tokens):
            Answers_tf = Answers_counter[Answers_token]/Answers_words_count
            Answers_df = doc_freq(Answers_token, Answers_DF)
            Answers_idf = np.log(Answers_N/(Answers_df+1))
            Answers_tf_idf[i, Answers_token] = Answers_tf*Answers_idf

    # Document Vectorization
    Answers_vectors = np.zeros((Answers_N, len(vocab)))
    for i in Answers_tf_idf:
        if(i[1] in vocab):
            Answers_ind = np.where(vocab == i[1]) 
            Answers_vectors[i[0]][Answers_ind] = Answers_tf_idf[i]


    cEXTAverage, cNEUAverage, cAGRAverage, cCONAverage, cOPNAverage = predictModel(Answers_vectors)

    PersonalityResults = {}
    PersonalityResults['cEXT'] = cEXTAverage
    PersonalityResults['cNEU'] = cNEUAverage
    PersonalityResults['cAGR'] = cAGRAverage
    PersonalityResults['cCON'] = cCONAverage
    PersonalityResults['cOPN'] = cOPNAverage
    return PersonalityResults

# Load dataset
#X_train, X_test, y_train, y_test = loadDataset()

# Preprocess training data
#X_train_dataset = preprocessDataset(X_train)

# Preprocess testing data
#X_test_dataset = preprocessDataset(X_test) 

# Calculate document frequency for training data
#X_train_DF = calculateDF(X_train_dataset)

# Get total vocabulary
#total_vocab = [x for x in X_train_DF]
#np.save('vectors/total_vocab.npy', total_vocab)
total_vocab = np.load('vectors/total_vocab.npy')

# Calculate document frequency for testing data
#X_test_DF = calculateDF(X_test_dataset)

# Calculate term frequency inverse document frequency for training data
#X_train_TFIDF = calculateTFIDF(X_train_dataset, X_train_DF)

# Calculate term frequency inverse document frequency for testing data
#X_test_TFIDF = calculateTFIDF(X_test_dataset, X_test_DF)

# Vectorize training data
#train_x_vectors = trainingDataVectorization(X_train_dataset, X_train_TFIDF, total_vocab)
#np.save('vectors/train_x_vectors.npy', train_x_vectors)
#train_x_vectors = np.load('vectors/train_x_vectors.npy')

# Vectorize testing data
#test_x_vectors = testingDataVectorization(X_test_dataset, X_test_TFIDF, total_vocab)
#np.save('vectors/test_x_vectors.npy', test_x_vectors)
#test_x_vectors = np.load('vectors/test_x_vectors.npy')

#train_y_cEXT, train_y_cNEU, train_y_cAGR, train_y_cCON, train_y_cOPN, test_y_cEXT, test_y_cNEU, test_y_cAGR, test_y_cCON, test_y_cOPN = prepareModel(y_train, y_test)

#clf_rf_cEXT, clf_rf_cNEU, clf_rf_cAGR, clf_rf_cCON, clf_rf_cOPN = trainModel(train_x_vectors, train_y_cEXT, train_y_cNEU, train_y_cAGR, train_y_cCON, train_y_cOPN)

#saveModel(clf_rf_cEXT, clf_rf_cNEU, clf_rf_cAGR, clf_rf_cCON, clf_rf_cOPN) 

Answers = ['I started my career in Marketing after graduating with a Business degree in 2013. I’ve spent my entire career at Microsoft, receiving two promotions and three awards for outstanding performance. I’m looking to join a smaller company now, and take on more leadership and project management.', 'From what I read, your company is one of the leaders in database and website security for large corporations. I read your list of clients on your website and saw multiple Fortune 500 companies mentioned, including Verizon and IBM. Beyond that, I recently had an informational interview with James from the Marketing team, after messaging him on LinkedIn, and he shared a bit about your company culture; mainly, the emphasis on collaboration and open interaction between different departments and groups. That’s something that sounds exciting to me and that I’m hoping to find in my next job. Can you share more about how you’d describe the company culture here?', 'I know you’re one of the leaders in contract manufacturing for the pharmaceutical industry. I read two recent news articles as well and saw that you just finalized plans to build a new facility that will double your manufacturing capacity. One of my hopes in my current job search is to find a fast-growing organization that could take full advantage of my past experience in scaling up manufacturing operations, so I was excited to have this interview and learn more about the specific work and challenges you need help with from the person you hire for this role.']

print(predictPersonality(Answers, total_vocab))