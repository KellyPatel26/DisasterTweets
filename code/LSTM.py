from preprocess import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gym import make
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt


### I need to modify the follwing code 

class LSTM_Seq2Dis(tf.keras.Model):
	def __init__(self, window_size, vocab_size):
		###### DO NOT CHANGE ##############
		super(LSTM_Seq2Dis, self).__init__()
		self.vocab_size = vocab_size 
		self.window_size = window_size 
		######^^^ DO NOT CHANGE ^^^##################
		def make_vars(*dim,initializer=tf.random.normal):
			return tf.Variable(initializer(dim,stddev=0.1))

		# TODO:
		# 1) Define any hyperparameters

		# Define batch size and optimizer/learning rate
		self.batch_size = 100 
		self.embedding_size = 300 
		self.learning_rate=0.01
		self.RNN=self.embedding_size
		self.dropRate=0.5

		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.embeddingVocab=make_vars(self.vocab_size, self.embedding_size)

		

		self.lstmdis=tf.keras.layers.LSTM(self.RNN,return_sequences=True,return_state=True,recurrent_dropout=0.3)
		self.drop=tf.keras.layers.Dropout(self.dropRate)
		# first type of dense
		self.denseTypeOne=tf.keras.layers.Dense(self.embedding_size,activation='linear')
		# second type of dense
		self.denseTypeTwo=tf.keras.layers.Dense(2,activation='softmax')
		# max pooling
		self.maxpool=tf.keras.layers.MaxPool1D()
		self.optimizer=tf.keras.optimizers.Adam(self.learning_rate)

	@tf.function
	def call(self, batched_input):
		# print('batched_input.shape',batched_input.shape)
		
		embededSentence=tf.nn.embedding_lookup(self.embeddingVocab,batched_input)
		sentence_output, final_memory_state,final_carry_state=self.lstmdis(embededSentence,None)
		denseOne=self.denseTypeOne(tf.concat([final_memory_state,final_carry_state],1))
		# denseTwo=self.denseTypeOne(denseOne)
		prbs=self.denseTypeTwo(denseOne)

		
		# prbs = tf.keras.layers.Flatten()(prbs)		
		return prbs

	def accuracy_function(self, prbs, labels):
		"""
		DO NOT CHANGE
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""
		decoded_symbols = tf.argmax(input=prbs, axis=1)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32))
		return accuracy


	def loss_function(self, prbs, labels):
		"""
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		lossTensor=tf.keras.losses.sparse_categorical_crossentropy(labels,prbs)
		loss=tf.math.reduce_sum(lossTensor) 	


		return loss

	


def train(model, X_token, y_train):
	
	iteration=int(tf.math.floor(len(y_train)/model.batch_size))
	losses = []
	for iter in range(iteration):		
		batchedInput=X_token[iter*model.batch_size:(iter+1)*model.batch_size]
		batchedLabel=y_train[iter*model.batch_size:(iter+1)*model.batch_size]
		
		with tf.GradientTape() as tape:
			batchedProbs=model.call(batchedInput)
			loss=model.loss_function(batchedProbs,batchedLabel)
			losses.append(loss)
		Grad=tape.gradient(loss,model.trainable_variables)
		model.optimizer.apply_gradients(zip(Grad,model.trainable_variables))
	return losses
	


def test(model, test_token, y_test):
	iteration=int(tf.math.floor(len(y_test)/model.batch_size))
	losses=[]
	for iter in range(iteration):
		batchedInput=test_token[iter*model.batch_size:(iter+1)*model.batch_size]
		batchedLabel=y_test[iter*model.batch_size:(iter+1)*model.batch_size]        
		probs = model.call(batchedInput)
		loss=model.loss_function(probs,batchedLabel)
		losses.append(loss)
	return losses


def main():
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
	window_size=20
	model=LSTM_Seq2Dis(window_size,len(X_token))
	
	train_losses = train(model,X_token,y_train)
	train_iters = np.arange(len(train_losses))

	test_losses=test(model,test_token,y_test)
	test_iters=np.arange(len(test_losses))

	test_prbs=model.call(test_token)
	test_acc=model.accuracy_function(test_prbs,y_test)
	print("Testing Accuracy:", test_acc)

	train_prbs=model.call(X_token)
	train_acc=model.accuracy_function(train_prbs,y_train)
	print("Training Accuracy:", train_acc)


 
	# plotting the points
	plt.figure(0)
	plt.plot(train_iters, train_losses)
	
	# naming the x axis
	plt.xlabel('Iterations')
	# naming the y axis
	plt.ylabel('Training Loss')
	
	# giving a title to my graph
	plt.title('Training Loss')
	
	# function to show the plot
	plt.show()

	plt.figure(1)
	plt.plot(test_iters, test_losses)
	
	# naming the x axis
	plt.xlabel('Iterations')
	# naming the y axis
	plt.ylabel('Testing Loss')
	
	# giving a title to my graph
	plt.title('Testing Loss')
	
	# function to show the plot
	plt.show()


	
if __name__ == '__main__':
	main()