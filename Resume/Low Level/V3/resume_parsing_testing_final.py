# -*- coding: utf-8 -*-
"""Resume Parsing - Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nj0GF9fZFydw6bi1QUe_9ph8jzuV-j3Z

# Resume Parsing
Version 3.0
Testing Module

# Imports
"""

# !pip install PyMuPDF
# !pip install docx2pdf
#!pip install pdfx

import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import fitz
from docx2pdf import convert
from collections import OrderedDict
import sys
import re
import pdfx
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') 
#sys.path.append(os.path.join(os.path.dirname(__file__), "Integration"))
 ## Getting Matrix for TFIDF for each word in all documents
file_path= 'train_tf_idf.csv'
train_df = pd.read_csv(file_path, index_col=0)

"""# Pre-processing
* It consists of some main steps
    * Lowercase
    * Removing Punctuation
    * Tokenization
    * Stopword Filtering
    * Stemming
    * Lemmatization

## Cleaning
"""

def clean(df):
  resume_df = df.copy()
  resume_df[['ID']] = resume_df[['ID']].apply(pd.to_numeric, errors='coerce')
  resume_df.drop(columns=['Resume_html'], inplace = True)
  return resume_df

"""## Lowercase"""

def to_lower(df):
  lower_df = df.copy()
  lower_df["Resume_str"] = lower_df["Resume_str"].str.lower()
  return lower_df

"""## Removing Punctuation"""

def rem_punct(df):
  punct_df = df.copy()
  punct_df['punct_sent'] = punct_df.apply(lambda row: "".join([char for char in row['Resume_str'] if char not in string.punctuation]), axis=1)
  return punct_df

"""## Tokenization"""

def to_tokens(df):
  tokens_df = df.copy()
  tokens_df['tokenized_sents'] = tokens_df.apply(lambda row: nltk.word_tokenize(row['punct_sent']), axis=1)
  return tokens_df

"""## Stop Words"""

def rem_stop_words(df):
  stop_df = df.copy()
  stop_words = stopwords.words('english')
  stop_df['stop_words'] = stop_df.apply(lambda row: [word for word in row['tokenized_sents'] if word not in stop_words], axis=1)
  return stop_df

"""## Stemming"""

def stemming(df):
  new_df = df.copy()
  porter = PorterStemmer()
  new_df['Stemmed'] = df.apply(lambda row:[porter.stem(word) for word in row['stop_words']], axis = 1)
  return new_df

"""## Lemmatization"""

def lemmatizing(df):
  new_df = df.copy()
  lemmatizer = WordNetLemmatizer()
  new_df['lemmatized'] = df.apply(lambda row:[lemmatizer.lemmatize(word) for word in row['stop_words']], axis = 1)
  return new_df

"""# Feature Extraction

## TF
"""

def TF_doc(df):
  TF_document_df = df.copy()
  #### Retrun to stemming / lemmetization
  TF_document_df['TF_doc'] = TF_document_df.apply(lambda row: Counter(row['stop_words']), axis=1)
  return TF_document_df

"""## TF for each Category"""

def TF_category(df):
  ## Getting the Categories Names
  TF_cat = pd.DataFrame(
                  columns=pd.Index( ["Test"]),
                  index=pd.Index([]))
  ## Calculating words frequency within each category
  for index, row in df.iterrows():
    for item, value in row["TF_doc"].items():
      if item not in TF_cat.index:
        TF_cat.loc[item] = 0
        TF_cat.loc[item, "Test"] = value
      else:      
        TF_cat.loc[item, "Test"] += value

  return TF_cat

def TF_Normalize(df):
  TF_Normalized = df.copy()
  for col in TF_Normalized:
    TF_Normalized[col]/=TF_Normalized[col].sum()
  return TF_Normalized

"""## IDF"""

def IDF(df):
  # Make a Copy to work with
  IDF_DF = df.copy()
  
  # 1+ loge(No of documents in corpus/No. of documents containing the word)
  IDF_DF['IDF'] = IDF_DF.apply(lambda row: 1+ np.log(len(IDF_DF.columns)/row.astype(bool).sum()), axis=1)

  return IDF_DF

"""## TF-IDF"""

def TF_IDF(TF, IDF):
  TF_IDF_DF = TF.copy()
  TF_IDF_DF = TF_IDF_DF.multiply(IDF["IDF"], axis="index")
  return TF_IDF_DF

"""## Overview"""

def overview_extraction(inp_txt, pdf_file):
  #print("Debug input:", inp_txt)
  out_dict = dict()
  ext_email = list()
  ext_links = list()
  phone_no = list()

  ## Email 
  mail_pattern= r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
  mail_search = re.findall(mail_pattern, inp_txt, re.M|re.I)
  if(mail_search):
    ext_email = mail_search
    #print("Found Email:", mail_search)

  ## Phone 
  phone_pattern = r"[0-9]{10}[0-9]{0,2}"
  phone_search = re.findall(phone_pattern, inp_txt, re.I|re.M)
  if(phone_search):
    phone_no = phone_search

  ## Links
  links_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+" 
  links_search = re.findall(links_pattern, inp_txt, re.M|re.I)
  if(links_search):
    ext_links = links_search
    #print("Normal links:", links_search)

  ## Shortened links
  # short_links_pattern = r"(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:com|net|edu|gov|mil|org)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))*" 
  # short_links_search = re.findall(short_links_pattern, inp_txt, re.M|re.I)
  # if short_links_search:
  #   for item in short_links_search:
  #     if item not in ext_email and item not in ext_links:
  #       ext_links.append(item)
  #   print("Found Short Link:", short_links_search)

  ## references
  #print("EMail list type:", type(ext_email))
  #print("Check:", 'maram311999@gmail.com' not in ext_email)
  try:
    for link in pdf_file.get_references_as_dict()['url']:
      if (link not in ext_links) and not link.startswith('mailto:'):
        ext_links.append(link)
  except:
    print("No URLs found!")

  out_dict["Phone"] = phone_no
  out_dict["Email"] = ext_email
  out_dict["Links"] = ext_links
  return out_dict

"""# Pipeline"""

def pipeline(file_name):
  resume_df = pd.read_csv(file_name)
  cleaned_df = clean(resume_df)
  lowered_df = to_lower(cleaned_df)
  punct_df = rem_punct(lowered_df)
  tokenized_df = to_tokens(punct_df)
  stop_words_df = rem_stop_words(tokenized_df)
  stemmed_df = stemming(stop_words_df)
  lemma_df = lemmatizing(stemmed_df)
  TF_doc_df = TF_doc(lemma_df)
  TF_cat_df = TF_category(TF_doc_df)
  TF_Norm_df = TF_Normalize(TF_cat_df)
  IDF_df = IDF(TF_cat_df)
  TF_IDF_df = TF_IDF(TF_Norm_df, IDF_df)

  return TF_IDF_df

"""## Cosine Similarity"""

def cosine_similarity(vec1, vec2):
  return (np.dot(vec1,vec2))/(np.sqrt(sum(np.square(vec1)))*np.sqrt(sum(np.square(vec2))))

"""# Main"""

def read_pdf(fname):
  if fname.split(".")[1] == 'docx':
        convert(fname)
        fname = fname.split(".")[0] + ".pdf"
  elif fname.split(".")[1] == 'pdf':
      pass
  else:
      print("Only PDF and docx types are supported!")
      return
  
  doc = pdfx.PDFx(fname)

  text = doc.get_text()
  # doc = fitz.open(fname)
  # text = ""
  # for page in doc:
  #     text = text + str(page.get_text())

  return text, doc

def main(file_name):
  ## Building Query TFIDF Vector
  sample_df = pd.read_csv(file_name)
  TF_IDF_test = pipeline(file_name)

 
  ## Getting the Categories Names
  TF_cat = pd.DataFrame(
                  columns=pd.Index(train_df.columns),
                  index=pd.Index([]))
  
  #for item in TF_IDF_test
  for index, row in TF_IDF_test.iterrows():
    if index not in train_df.index:
      TF_cat.loc[index] = 0
    else:      
      TF_cat.loc[index] = train_df.loc[index]

  scores_dict = dict()
  ## Calculating Similarity
  TF_IDF_arr_query = np.array(TF_IDF_test["Test"])
  max_score = -1
  cat_sim = ""
  #print("Similarities:#########")
  for col in TF_cat.columns:
    temp_arr = np.array(TF_cat[col])
    sim_score = cosine_similarity(TF_IDF_arr_query, temp_arr)
    scores_dict[col] = sim_score
    if sim_score> max_score:
      max_score = sim_score
      cat_sim = col
    #print(col, sim_score)

  print("########\nWinning Category:", cat_sim, "\nWith Score:", max_score)

  scores_dict = sorted(scores_dict.items(), key=lambda item: item[1], reverse= True)
  sorted_dict = dict()
  for k, v in scores_dict:
      sorted_dict[k] = v

  return TF_IDF_test, train_df, TF_cat, sorted_dict

def test_func(file_name):
  ID = file_name.split(".")[0]
  my_txt= ""
  output_dict = dict()
  overview_dict = dict()
  try:
    my_txt, pdf_file= read_pdf(file_name)
  except:
    print("No such file!")
    return output_dict
  build_df = pd.DataFrame(
                  columns=pd.Index(['ID',"Resume_str", 'Resume_html']),
                  index=pd.Index([]))
  dict_str = {'ID': ID, "Resume_str": my_txt, 'Resume_html': my_txt}
  build_df = build_df.append(dict_str, ignore_index = True)
  build_df.to_csv(ID+".csv")
  TF_IDF_vec_query, weights_df, TF_IDF_vec_train, scores_dict = main(ID+".csv")
  overview_dict = overview_extraction(my_txt, pdf_file)
  output_dict["Ranking"] = scores_dict
  output_dict["Overview"] = overview_dict
  print(output_dict)
  return output_dict



#if __name__ == '__main__':
#  globals()[sys.argv[1]](sys.argv[2])

