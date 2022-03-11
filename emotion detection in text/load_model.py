import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import classification_report
from transformers import AutoTokenizer,TFBertModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')
model_weights = 'sentiment_weights.h5'
max_len = 70


input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

def load_model(texts):
    embeddings = bert(input_ids,attention_mask = input_mask)[0] 
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = Dense(128, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = Dense(32,activation = 'relu')(out)
    y = Dense(6,activation = 'sigmoid')(out)       
    new_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    new_model.layers[2].trainable = True
    new_model.load_weights('sentiment_weights.h5')

    x_val = tokenizer(
        text=texts,
        add_special_tokens=True,
        max_length=70,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True) 
    validation = new_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    return validation