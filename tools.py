import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from sklearn import preprocessing
import re
from pickle import dump, load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PredTools:
    params = {
        "num_classes" : 53,
        "epochs" : 1000,
        "batch_size": 64,
        "feature" : 5000,
        "embedding_size" : 200,
        "maxlen" : 22, # Max Lenth of sentence to pad to
        "trunc_type" : 'post',
        "padding_type" : 'pre',
        "oov_tok" : '<OOV>'
        }
   
    def read_and_transform_data(self, data_path):
        #read data
        asd_df = pd.read_excel(data_path)
        return asd_df
    
    #encode lables
    def encode_lables(self, data_path):
        data_frame = self.read_and_transform_data(data_path)
        le = preprocessing.LabelEncoder()
        le.fit(data_frame['Groups'])
        data_frame['Target'] = le.transform(data_frame['Groups'])
        return data_frame, le
    #decode lables
    def decode_lables(self, data_path, encoded_group):
        encoder = self.encode_lables(data_path)[1]
        decoded_group = encoder.inverse_transform([encoded_group])
        return decoded_group

    #data_path
    def tokenizer(self,data_path):
        feature=self.params['feature']
        oov_tok=self.params['oov_tok']
        tokenizer = Tokenizer(num_words=feature, oov_token=oov_tok)
        asd_df_prep = self.read_and_transform_data(data_path)
        tokenizer.fit_on_texts([str(x) for x in list(asd_df_prep['cleanText'])])
        word_index = tokenizer.word_index
        return tokenizer
    #sequence and pad tokenized data
    def pad_and_sequence_data(self, data_path, data):
        #params
        maxlen = self.params['maxlen']
        padding_type = self.params['padding_type']
        trunc_type = self.params['trunc_type']
        
        tokenizer = self.tokenizer(data_path)
        seq_text = tokenizer.texts_to_sequences([data])
        padded_seq = pad_sequences(seq_text,maxlen = maxlen, padding=padding_type, truncating=trunc_type)
        return padded_seq

    # predect a value
    def predict(self, data_path, path_to_weight, input_text):
        #model = self.create_model(data_path)
        model = load(open('model.pkl','rb'))
        model.load_weights(path_to_weight)
        clean_input = input_text
        paded_input_text = self.pad_and_sequence_data(data_path, clean_input)
        predicted_encoded_index = np.argmax(model.predict(paded_input_text))
        predicted_decoded_group = self.decode_lables(data_path,predicted_encoded_index)
        return predicted_decoded_group