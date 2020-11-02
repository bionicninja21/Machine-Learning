import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Embedding
from keras.initializers import Constant

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

from keras.models import load_model
model = load_model('C:/Users/bioni/PycharmProjects/quora/my_model.h5')
print('Model Loaded')

embeddings_index = {}
f = open('C:/Users/bioni/PycharmProjects/quora/glove.840B.300d/glove.840B.300d.txt',encoding="utf-8")
for line in f:
    values = line.split(' ')
    word = values[0] ## The first entry is the word
    coefs = np.asarray(values[1:], dtype='float32') ## These are the vectors representing the embedding for the word
    embeddings_index[word] = coefs
f.close()

print('GloVe data loaded')

train= pd.read_csv('C:/Users/bioni/PycharmProjects/quora/train.csv')
test=train[train['target']==1].head(10)

line='When should I apply for RV college of engineering and BMS college of engineering? Should I wait for the COMEDK result or am I supposed to apply before the result?'
test.loc[0] = ['testing'  , line]
test.loc[1]=['t2','Why does velocity affect time?']
test.loc[2]=['t3','I am gay boy and I love my cousin (boy).']
test.loc[3]=['t4','Which races have the smallest penis?']	
test.loc[4]=['t5','How were the Calgary Flames founded?']

'''## Iterate over the data to preprocess by removing stopwords
lines_without_stopwords=[] 
for line in test['question_text'].values: 
    line = line.lower()
    line_by_words = re.findall(r'(?:\w+)', line, flags = re.UNICODE) # remove punctuation and split
    new_line=[]
    for word in line_by_words:
        if word not in stop:
            new_line.append(word)
    lines_without_stopwords.append(new_line)
texts = lines_without_stopwords

print(texts[0:5])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

MAX_NUM_WORDS = 1000
MAX_SEQUENCE_LENGTH = 100
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print(data.shape)


from keras.layers import Embedding
from keras.initializers import Constant

EMBEDDING_DIM = embeddings_index.get('a').shape[0]
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

from keras.models import load_model
model = load_model('C:/Users/bioni/PycharmProjects/quora/my_model.h5')
print('Model Loaded')

pred=model.predict(data, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

pred = np.round(pred)
pred = pred.astype(int)
pred=np.delete(pred,1,1)
df2 = pd.DataFrame(data=pred)

print(pred[0,0])
df2[df2[0]==0].head(20)
'''

from tkinter import *
from tkinter import messagebox
window=Tk()

lbl=Label(window,text="Please enter your question below")
lbl.place(relx=0.3, rely=0.5, height=26, width=238)
Label2=Label(window,text='QUORA QUESTION CLASSIFIER',font='Helvetica 16 bold')
Label2.place(relx=0.133, rely=0.22, height=107, width=457)
txtfld=Entry(window)

btn=Button(window, text="Go",command=clicked)

btn.place(relx=0.417, rely=0.8, height=33, width=96)

def clicked():
 
    line=txtfld.get()
    line = line.lower()
    lines_without_stopwords=[] 
    line_by_words = re.findall(r'(?:\w+)', line, flags = re.UNICODE) # remove punctuation and split
    new_line=[]
    for word in line_by_words:
        if word not in stop:
            new_line.append(word)
    lines_without_stopwords.append(new_line)
    texts = lines_without_stopwords
    
    MAX_NUM_WORDS = 1000
    MAX_SEQUENCE_LENGTH = 100
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
   
    pred=model.predict(data, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    pred = np.round(pred)
    pred = pred.astype(int)
    pred=np.delete(pred,0,1)

    print(pred[0,0])

    if (pred[0,0]==0):
        messagebox.showwarning("Prediction","This question is Valid")
    else:
        messagebox.showinfo("Prediction","This question is Invalid")
txtfld.place(relx=0.317, rely=0.58,height=84, relwidth=0.357)
window.title('Quora Insincere Question Classifier')
window.geometry("600x500")
window.mainloop()

def find():
    t=Entry.get()
    print(t)
    
