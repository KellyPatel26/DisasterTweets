from preprocess import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':

    # Preprocessing
    X_train, y_train, X_test, y_test = get_data()
    max_features = 3000
    '''
    X_token is the tokenized training text
    test_token is the tokenized testing text
    '''
    # tokenize training
    tokenizer = Tokenizer(num_words = max_features, split=" ")
    tokenizer.fit_on_texts(X_train.values)
    X_token = tokenizer.texts_to_sequences(X_train.values)
    X_token = pad_sequences(X_token) 
    # tokenize testing
    tokenizer.fit_on_texts(X_train.values)
    test_token = tokenizer.texts_to_sequences(X_test.values)
    test_token = pad_sequences(test_token)
    # print(tokenizer.sequences_to_texts([X_token[0]]))

    