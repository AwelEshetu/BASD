from flask import Flask, render_template,request
from tools import PredTools
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

@app.route('/', methods=["GET"])
def index():
   return render_template('index.html')

@app.route('/', methods=["POST"])
def predict():
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
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    email = request.form['email']
    subject = request.form['subject']
    tool = PredTools()
    prediction = tool.predict('cleaned_input_data.xlsx','weights.bestRANDOM.hdf5', subject)
    group = prediction[0]
    return render_template('index.html', prediction=group)
if __name__=='__main__':
    app.run()