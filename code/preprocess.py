import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import re
import string 



stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
            'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
            'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 
             'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
              'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
               'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
               'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
                'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
                 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 
                 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                  "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
                  'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
def balance(data):
    '''
    This function balances the data so that around 50% are labeled 1 and 50% are labeled 0.
    We will use undersampling.

    inputs:
        data - a dataframe of a csv with an text column and a target column
    output:
        a balanced dataset
    '''
    data_0_class = data[data['target'] == 0]
    data_1_class = data[data['target'] == 1]
    data_0_class_undersampled = data_0_class.sample(data_1_class.shape[0])
    data = pd.concat([data_0_class_undersampled, data_1_class], axis = 0)
    return data

def remove_URLS(text):
    '''
    Removes URLS from text
    '''
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r' ', text)


# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    '''
    Removes emojis from text. Used reference above.
    Example:
        text = "Sad days 😔😔"
        remove_emoji("Sad days 😔😔") = Sad days
    '''
    emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    return emojis.sub(r' ', text)

def remove_punct(text):
    '''
    Removes punctuation from text including hashtags
    Example:
        text = "#Newswatch 2 vehicles"
        output = "Newswatch 2 vehicles"
    '''
    table = str.maketrans(' ',' ', string.punctuation)
    return text.translate(table)

def remove_stopwords(text):
    '''
    Removes stopwords like "to, no, for,..." to increase accuracy and decrease runtime
    '''
    remove_stopword = [word for word in text.split() if word.lower() not in stopwords]
    return remove_stopword


def get_data():
    '''
    Returns (in this order):
        - X_train (4906,) of training text
        - Y_train (4906,) of training labels
        - X_test (1636,) of testing text
        - Y_test (1636,) of testing labels
    '''

    df = pd.read_csv("../data/train.csv", encoding="ISO-8859-1")
    # Dropping id, keyword, and location columns
    df = df.drop(['id', 'keyword', 'location'], axis = 1)
    # There are 4342 0 labels
    # There are 3271 1 labels
    # we need to even this out
    df = balance(df)
    # remove URLS
    df['text'] = df['text'].apply(lambda x: remove_URLS(x))
    # remove emojis
    df['text'] = df['text'].apply(lambda x: remove_emoji(x))
    # remove punctuation
    df['text'] = df['text'].apply(lambda x: remove_punct(x))
    # make everything lowercase
    df['text'] = df['text'].apply(lambda x: str.lower(x))
    # remove stop words:
    # df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

    # Remove multiple spaces
    df['text']= df['text'].str.replace('   ', ' ')
    df['text']= df['text'].str.replace('     ', ' ')
    df['text']= df['text'].str.replace('\xa0 \xa0 \xa0', ' ')
    df['text']= df['text'].str.replace('  ', ' ')
    df['text']= df['text'].str.replace('—', ' ')
    df['text']= df['text'].str.replace('-', ' ')

    print(df.head(10))
    # split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(df['text'],df['target'], stratify=df['target'])
    
    return X_train, y_train, X_test, y_test
