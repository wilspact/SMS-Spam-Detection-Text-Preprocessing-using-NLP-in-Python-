import pandas as pd 
data=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\dataset\\spam.csv")
print(data.head())
data=data[['Category','Message']]
print(data.head())
print(data['Category'].value_counts())
from string import punctuation
def remove_punctuation(text):
    punct="".join([i for i in text if i not in punctuation])
    return punct
data['clean_msg']=data['Message'].apply(lambda x:remove_punctuation(x))
print(data.head())
data['msg_lower']=data['clean_msg'].apply(lambda x:x.lower())
print(data.head())
import re 
def tokenization(text):
    tokens=re.split('W+',text)
    return tokens 
data['msg_tokenization']=data['msg_lower'].apply(lambda x:tokenization(x))
print(data.head())
import nltk
Stopwords=nltk.corpus.stopwords.words("english")
def remove_stopwords(text):
    stop=[word for word in text if word not in Stopwords ]
    return stop
data['no_stopwords']=data['msg_tokenization'].apply(lambda x:remove_stopwords(x))
print(Stopwords)
print(data[['Message','no_stopwords']].head())
from nltk.stem.snowball import SnowballStemmer
stemobj=SnowballStemmer("english")
stemmed=[]
def stemming(text):
    stem_text=[stemobj.stem(word)for word in text]
    return stem_text
data['msg_stemmed']=data['no_stopwords'].apply(lambda x:stemming(x))
print(data.head())
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_lem=WordNetLemmatizer()
word_lem.lemmatize("english")
def lemmatizing(text):
    lem_text=[word_lem.lemmatize(word) for word in text]
    return lem_text
data['msg_lemmitized']=data['no_stopwords'].apply(lambda x:lemmatizing(x))
print(data.head())
