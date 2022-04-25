from lib2to3.pgen2 import token
from preprocess import *
import tensorflow as tf
import transformers

#######################
# Some useful variables
MAX_SEQUENCE_LENGTH = 512
MODEL_NAME = 'distilbert-base-uncased'
EMBEDDING_SIZE = 768
LEARNING_RATE = 0.01
BATCH_SIZE = 16
#######################



if __name__ == '__main__':
    # Preprocessing
    X_train, y_train, X_test, y_test = get_data()

    # Get tokenizer
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Tokenizing training and testing input and putting all inputs and labels 
    # into proper format
    tokenized_train_input = tokenizer(
        X_train.values.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="tf"
    )

    tokenized_test_input = tokenizer(
        X_test.values.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="tf"
    )

    train_input = dict(tokenized_train_input)
    train_label = tf.convert_to_tensor(y_train)
    test_input = dict(tokenized_test_input)
    test_label = tf.convert_to_tensor(y_test)


    # Get pretrained BERT weights
    bert = transformers.TFDistilBertModel.from_pretrained(
      MODEL_NAME,
      num_labels=2
      )
    
    # We are using feature based approach, so we are not going to 
    # further train the BERT model
    for layer in bert.layers:
      layer.trainable = False
    
    # Initializing input, attention mask for BERT
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')
    
    # Getting the output from bert
    output = bert([input_ids, attention_mask]).last_hidden_state[:, 0, :]
    
    # Classification Layers
    classification_layers = tf.keras.Sequential(
      [
        tf.keras.layers.Dense(
          units=EMBEDDING_SIZE,
          activation='relu',
          name="dense_01",),
        tf.keras.layers.Dense(
          units=EMBEDDING_SIZE,
          activation='relu',
          name="dense_02",),
        tf.keras.layers.Dense(
          units=2,
          activation='softmax',
          name="softmax")
      ])

    # Feeding the output of BERT into classification layers 
    output = classification_layers(output)

    # Compiling the model and specifying loss, optimizer
    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
      metrics=['accuracy'])

    # Training the model
    model.fit(train_input,train_label,batch_size=BATCH_SIZE)

    # Testing the model
    test_loss, test_acc = model.evaluate(test_input,test_label)

    print('\nTest accuracy: {}'.format(test_acc))
    
    
    
    
    # If you need tokenization, see how I did it in LSTM.py!