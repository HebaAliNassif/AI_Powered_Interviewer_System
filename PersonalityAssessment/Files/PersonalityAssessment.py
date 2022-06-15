import numpy as np
import pandas as pd

import glob
import nltk 
import num2words 
import re
import pickle

import time


from sklearn.model_selection import train_test_split
from nltk import word_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import svm

start= time.time()
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
total_vocab = np.load("../../PersonalityAssessment/Files/vectors/total_vocab.npy")
end = time.time()
print("loading Personality Assessment Model takes: "+str(end - start)+" secs")

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

    return X_essays, y_essays

def splitDataset(): 
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

def prepareModel(y):
    y_cEXT = y.iloc[:, 0]
    y_cNEU = y.iloc[:, 1]
    y_cAGR = y.iloc[:, 2]
    y_cCON = y.iloc[:, 3]
    y_cOPN = y.iloc[:, 4]

    return y_cEXT, y_cNEU, y_cAGR, y_cCON, y_cOPN

def trainModel(train_x_vectors, train_y_cEXT, train_y_cNEU, train_y_cAGR, train_y_cCON, train_y_cOPN):

    print("training Extraversion cEXT using SVM...")
    clf_svm_cEXT = svm.SVC(probability=True)
    clf_svm_cEXT.fit(train_x_vectors, train_y_cEXT)
    #print("cEXT score: ", clf_svm_cEXT.score(test_x_vectors, test_y_cEXT))

    print("training Neuroticism cNEU using SVM...")
    clf_svm_cNEU = svm.SVC(probability=True)
    clf_svm_cNEU.fit(train_x_vectors, train_y_cNEU)
    #print("cNEU score: ", clf_svm_cNEU.score(test_x_vectors, test_y_cNEU))

    print("training Agreeableness cAGR using using SVM...")
    clf_svm_cAGR = svm.SVC(probability=True)
    clf_svm_cAGR.fit(train_x_vectors, train_y_cAGR)
    #print("cAGR score: ", clf_svm_cAGR.score(test_x_vectors, test_y_cAGR))

    print("training Conscientiousness cCON using SVM...")
    clf_svm_cCON = svm.SVC(probability=True)
    clf_svm_cCON.fit(train_x_vectors, train_y_cCON)
    #print("cCON score: ", clf_svm_cCON.score(test_x_vectors, test_y_cCON))
 
    print("training Openness to Experience cOPN using SVM...")
    clf_svm_cOPN = svm.SVC(probability=True)
    clf_svm_cOPN.fit(train_x_vectors, train_y_cOPN)
    #print("cOPN score: ", clf_svm_cOPN.score(test_x_vectors, test_y_cOPN))
    
    saveModel(clf_svm_cEXT, clf_svm_cNEU, clf_svm_cAGR, clf_svm_cCON, clf_svm_cOPN) 

    return clf_svm_cEXT, clf_svm_cNEU, clf_svm_cAGR, clf_svm_cCON, clf_svm_cOPN


def saveModel(clf_svm_cEXT, clf_svm_cNEU, clf_svm_cAGR, clf_svm_cCON, clf_svm_cOPN):  
    SVM_cEXT_model = 'models/SVM_cEXT_model.sav'
    pickle.dump(clf_svm_cEXT, open(SVM_cEXT_model, 'wb'))

    SVM_cNEU_model = 'models/SVM_cNEU_model.sav'
    pickle.dump(clf_svm_cNEU, open(SVM_cNEU_model, 'wb'))

    SVM_cAGR_model = 'models/SVM_cAGR_model.sav'
    pickle.dump(clf_svm_cAGR, open(SVM_cAGR_model, 'wb'))

    SVM_cCON_model = 'models/SVM_cCON_model.sav'
    pickle.dump(clf_svm_cCON, open(SVM_cCON_model, 'wb'))

    SVM_cOPN_model = 'models/SVM_cOPN_model.sav'
    pickle.dump(clf_svm_cOPN, open(SVM_cOPN_model, 'wb'))


def loadModel():
    SVM_cEXT_model = '../../PersonalityAssessment/Files/models/SVM_cEXT_model.sav'
    clf_svm_cEXT = pickle.load(open(SVM_cEXT_model, 'rb'))

    SVM_cNEU_model = '../../PersonalityAssessment/Files/models/SVM_cNEU_model.sav'
    clf_svm_cNEU = pickle.load(open(SVM_cNEU_model, 'rb'))
    
    SVM_cAGR_model = '../../PersonalityAssessment/Files/models/SVM_cAGR_model.sav'
    clf_svm_cAGR = pickle.load(open(SVM_cAGR_model, 'rb'))

    SVM_cCON_model = '../../PersonalityAssessment/Files/models/SVM_cEXT_model.sav'
    clf_svm_cCON = pickle.load(open(SVM_cCON_model, 'rb'))

    SVM_cOPN_model = '../../PersonalityAssessment/Files/models/SVM_cOPN_model.sav'
    clf_svm_cOPN = pickle.load(open(SVM_cOPN_model, 'rb'))

    return clf_svm_cEXT, clf_svm_cNEU, clf_svm_cAGR, clf_svm_cCON, clf_svm_cOPN
    


def predictModel(Answers_vectors):
    clf_svm_cEXT, clf_svm_cNEU, clf_svm_cAGR, clf_svm_cCON, clf_svm_cOPN = loadModel()

    cEXT = clf_svm_cEXT.predict_proba(Answers_vectors)
    cEXT = np.array(cEXT)
    cEXTAverage = (cEXT.sum(axis = 0)[1]*100)/len(cEXT)

    cNEU = clf_svm_cNEU.predict_proba(Answers_vectors)
    cNEU = np.array(cNEU)
    cNEUAverage = (cNEU.sum(axis = 0)[1]*100)/len(cNEU)

    cAGR = clf_svm_cAGR.predict_proba(Answers_vectors)
    cAGR = np.array(cAGR)
    cAGRAverage = (cAGR.sum(axis = 0)[1]*100)/len(cAGR)

    cCON = clf_svm_cCON.predict_proba(Answers_vectors)
    cCON = np.array(cCON)
    cCONAverage = (cCON.sum(axis = 0)[1]*100)/len(cCON)

    cOPN = clf_svm_cOPN.predict_proba(Answers_vectors)
    cOPN = np.array(cOPN)
    cOPNAverage = (cOPN.sum(axis = 0)[1]*100)/len(cOPN)

    return cEXTAverage, cNEUAverage, cAGRAverage, cCONAverage, cOPNAverage


def predictPersonality(path):
    Answers = []
    for filename in glob.glob(path + '/*.*'):
        #.append(filename)
        with open(filename, 'r') as content_file:
            data = content_file.readlines()
            Answers.append(data[0])

    Answers = np.asarray(Answers)
      
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
    Answers_vectors = np.zeros((Answers_N, len(total_vocab)))
    for i in Answers_tf_idf:
        if(i[1] in total_vocab):
            Answers_ind = np.where(total_vocab == i[1]) 
            Answers_vectors[i[0]][Answers_ind] = Answers_tf_idf[i]


    cEXTAverage, cNEUAverage, cAGRAverage, cCONAverage, cOPNAverage = predictModel(Answers_vectors)

    PersonalityResults = {}
    PersonalityResults['cEXT'] = cEXTAverage
    PersonalityResults['cNEU'] = cNEUAverage
    PersonalityResults['cAGR'] = cAGRAverage
    PersonalityResults['cCON'] = cCONAverage
    PersonalityResults['cOPN'] = cOPNAverage
    return PersonalityResults

# Load and split dataset
#X_train, X_test, y_train, y_test = splitDataset()

# Preprocess training data
#X_train_dataset = preprocessDataset(X_train)

# Preprocess testing data
#X_test_dataset = preprocessDataset(X_test) 

# Calculate document frequency for training data
#X_train_DF = calculateDF(X_train_dataset)

# Get total vocabulary
#total_vocab = [x for x in X_train_DF]
#np.save('vectors/total_vocab.npy', total_vocab)


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

#train_y_cEXT, train_y_cNEU, train_y_cAGR, train_y_cCON, train_y_cOPN = prepareModel(y_train)
#test_y_cEXT, test_y_cNEU, test_y_cAGR, test_y_cCON, test_y_cOPN = prepareModel(y_test)





#X_essays, y_essays = loadDataset()

#X_essays_dataset = preprocessDataset(X_essays)

#X_essays_DF = calculateDF(X_essays_dataset)

#total_vocab = [x for x in X_essays_DF]
#np.save('vectors/total_vocab.npy', total_vocab)
total_vocab = np.load('vectors/total_vocab.npy')


#X_essays_TFIDF = calculateTFIDF(X_essays_dataset, X_essays_DF)

#essays_x_vectors = trainingDataVectorization(X_essays_dataset, X_essays_TFIDF, total_vocab)
#np.save('vectors/essays_x_vectors.npy', essays_x_vectors)
#train_x_vectors = np.load('vectors/essays_x_vectors.npy')

#essays_y_cEXT, essays_y_cNEU, essays_y_cAGR, essays_y_cCON, essays_y_cOPN = prepareModel(y_essays)

#clf_svm_cEXT, clf_svm_cNEU, clf_svm_cAGR, clf_svm_cCON, clf_svm_cOPN = trainModel(train_x_vectors, essays_y_cEXT, essays_y_cNEU, essays_y_cAGR, essays_y_cCON, essays_y_cOPN)


path = "C:/Users/maram/Documents/GitHub/AI_Powered_Interviewer_System/PersonalityAssessment/SpeechRecognitionOutput"
print(predictPersonality(path))