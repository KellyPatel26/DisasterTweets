from preprocess import *

if __name__ == '__main__':

    # Preprocessing
    X_train, y_train, X_test, y_test = get_data()
    
    # If you need tokenization, see how I did it in LSTM.py!