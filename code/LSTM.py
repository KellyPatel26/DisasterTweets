from preprocess import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gym import make
import numpy as np
import tensorflow as tf


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


### I need to modify the follwing code 

class LSTM_Seq2Dis(tf.keras.Model):
	def __init__(self, window_size, vocab_size):
		###### DO NOT CHANGE ##############
		super(LSTM_Seq2Dis, self).__init__()
		self.vocab_size = vocab_size 
		self.french_window_size = window_size 
		######^^^ DO NOT CHANGE ^^^##################
		def make_vars(*dim,initializer=tf.random.normal):
			return tf.Variable(initializer(dim,stddev=0.1))

		# TODO:
		# 1) Define any hyperparameters

		# Define batch size and optimizer/learning rate
		self.batch_size = 50 
		self.embedding_size = 50 
		self.learning_rate=5e-3
		self.RNN=self.embedding_size
		self.dropRate=0.5

		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.embeddingVocab=make_vars(self.vocab_size, self.embedding_size)
		self.lstmdis=tf.keras.layers.LSTM(self.RNN,return_sequences=True,return_state=True,recurrent_dropout=0.3)
		self.drop=tf.keras.layers.Dropout(self.dropRate)
		# first type of dense
		self.denseTypeOne=tf.keras.layers.Dense(self.vocab_size,activation='relu')
		# second type of dense
		self.denseTypeTwo=tf.keras.layers.Dense(self.vocab_size,activation='sigmoid')
		# max pooling
		self.maxpool=tf.keras.layers.MaxPool1D()
		self.optimizer=tf.keras.optimizers.Adam(self.learning_rate)

	@tf.function
	def call(self, batched_input):

		embededSentence=tf.nn.embedding_lookup(self.embeddingVocab,batched_input)
		sentence_output, final_memory_state,final_carry_state=self.lstmdis(embededSentence,None)
		poolingRe=self.maxpool(final_memory_state)
		denseOne=self.denseTypeOne(poolingRe)
		dropOne=self.drop(denseOne,training=True)
		denseTwo=self.denseTypeOne(dropOne)
		dropTwo=self.drop(denseTwo,training=True)
		prbs=self.denseTypeTwo(dropTwo)		
		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		lossTensor=tf.keras.losses.sparse_categorical_crossentropy(labels,prbs)*mask
		loss=tf.math.reduce_sum(lossTensor) 	


		return loss

    