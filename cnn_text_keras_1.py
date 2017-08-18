import datetime
import os,sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.engine import Input, Model

from tensorflow.contrib.keras.python.keras.utils import to_categorical
from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer,text_to_word_sequence
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.keras.python.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

d_now=datetime.datetime.now()
TEXT_DATA_DIR = "./data"
GLOVE_DIR = "./glove.6B"
SAVE_DIR= "./Save"+ str(int(d_now.timestamp()))
if not os.path.exists(SAVE_DIR): os.mkdir(SAVE_DIR)

MAX_NB_WORDS=180000 #словарь
MAX_NB_WORDS_IN_TEXT=5000 # Если больше то режем на куски
EMBEDDING_DIM=300
VALIDATION_SPLIT=0.05
NUM_ROWS_FROM_TEXT=5000
NUM_ROWS_SAVE_TO_TRAIN=1000
NUM_ROWS_SAVE_TO_VAL=int(NUM_ROWS_SAVE_TO_TRAIN*VALIDATION_SPLIT)
NUM_EPOCHS=7
filename="./training_text"
filename_v="./training_variants"


texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids





def read_data(filename,filename_v):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    labels_index['0'] = 0
    diction=dict()
    ret=dict()
    i=0
    with open(filename_v, 'r', encoding="utf-8") as fv:
        for l in fv:
            if i>0:
                data = tf.compat.as_str(l.strip()).split(',')
                diction[int(data[0])]=[data[1],data[2],int(data[3])]
            i += 1

    i = 0
    with open(filename, 'r', encoding="utf-8") as f:

        for line in f:

            if i>0 and len(line)>100:

                id = int(line[:line.find('||')])
                t_w = text_to_word_sequence(line[line.find('||') + 2:])
                if len(t_w)>MAX_NB_WORDS_IN_TEXT:
                    for text_i in range(0,len(t_w),MAX_NB_WORDS_IN_TEXT):
                        if text_i+MAX_NB_WORDS_IN_TEXT-1<len(t_w):
                            texts.append(' '.join(t_w[text_i:text_i+MAX_NB_WORDS_IN_TEXT-1]))
                            labels.append(diction[id][2])
                        elif len(t_w)-text_i>50:
                            texts.append(' '.join(t_w[text_i:len(t_w)]))
                            labels.append(diction[id][2])
                else:
                    texts.append(' '.join(t_w))
                    labels.append(diction[id][2])

                labels_index[str(diction[id][2])]=diction[id][2]
            i+=1
            if i>=NUM_ROWS_FROM_TEXT :
                break
    return texts,labels,labels_index


texts,labels,labels_index =read_data(filename,filename_v)

print(max([len(t)for t in texts]))
print(min([len(t)for t in texts]))




tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print(max([len(t)for t in sequences]))
print(min([len(t)for t in sequences]))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
MAX_SEQUENCE_LENGTH = data.shape[1]
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

np.save (os.path.join(SAVE_DIR, 'embeddings_index.npy'), embeddings_index)
np.save (os.path.join(SAVE_DIR, 'embedding_matrix.npy'), embedding_matrix)
np.save (os.path.join(SAVE_DIR, 'word_index.npy'), word_index)
np.save (os.path.join(SAVE_DIR, 'labels_index.npy'),labels_index)

lendata = data.shape[0]

i=0
for j in range(NUM_ROWS_SAVE_TO_TRAIN-NUM_ROWS_SAVE_TO_VAL, lendata, NUM_ROWS_SAVE_TO_TRAIN-NUM_ROWS_SAVE_TO_VAL ):
    np.save(os.path.join(SAVE_DIR, 'data_'+str(i)+'_'+str(j)+'.npy'), data[i:j])
    np.save(os.path.join(SAVE_DIR, 'labels_' + str(i) + '_' + str(j) + '.npy'), labels[i:j])
    np.save(os.path.join(SAVE_DIR, 'data_'+str(j+1)+'_'+str(j+NUM_ROWS_SAVE_TO_VAL)+'.npy'), data[j+1:j+NUM_ROWS_SAVE_TO_VAL])
    np.save(os.path.join(SAVE_DIR, 'labels_' + str(j+1) + '_' + str(j+NUM_ROWS_SAVE_TO_VAL) + '.npy'), labels[j+1:j+NUM_ROWS_SAVE_TO_VAL])
    i=j+1
np.save(os.path.join(SAVE_DIR, 'data_'+str(i)+'_'+str(lendata)+'.npy'), data[i:lendata])
np.save(os.path.join(SAVE_DIR, 'labels_' + str(i) + '_' + str(lendata) + '.npy'), labels[i:lendata])
np.save(os.path.join(SAVE_DIR, 'data_' + str(lendata-1) + '_' + str(lendata) + '.npy'), data[lendata-1:lendata])
np.save(os.path.join(SAVE_DIR, 'labels_' + str(lendata-1) + '_' + str(lendata) + '.npy'), labels[lendata-1:lendata])

tokenizer = None
data= None
labels=None

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 15, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

embedding_matrix=None
embeddings_index=None

for epoch in range(NUM_EPOCHS):
    i=0
    for j in range(NUM_ROWS_SAVE_TO_TRAIN-NUM_ROWS_SAVE_TO_VAL, lendata, NUM_ROWS_SAVE_TO_TRAIN-NUM_ROWS_SAVE_TO_VAL ):
        x_train = np.load(os.path.join(SAVE_DIR, 'data_'+str(i)+'_'+str(j)+'.npy'))
        y_train = np.load(os.path.join(SAVE_DIR, 'labels_' + str(i) + '_' + str(j) + '.npy'))
        x_val = np.load(os.path.join(SAVE_DIR, 'data_'+str(j+1)+'_'+str(j+NUM_ROWS_SAVE_TO_VAL)+'.npy'))
        y_val = np.load(os.path.join(SAVE_DIR, 'labels_' + str(j+1) + '_' + str(j+NUM_ROWS_SAVE_TO_VAL) + '.npy'))
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=100)
        model.save(os.path.join(SAVE_DIR,'CNN_woVEC'))
        i=j+1
        print("---------Epoch ",epoch,"   ------batch  ",i)
    x_train = np.load(os.path.join(SAVE_DIR, 'data_'+str(i)+'_'+str(lendata)+'.npy'))
    y_train = np.load(os.path.join(SAVE_DIR, 'labels_' + str(i) + '_' + str(lendata) + '.npy'))
    x_val = np.load(os.path.join(SAVE_DIR, 'data_' + str(lendata-1) + '_' + str(lendata) + '.npy'))
    y_val = np.load(os.path.join(SAVE_DIR, 'labels_' + str(lendata-1) + '_' + str(lendata) + '.npy'))
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=100)
    model.save(os.path.join(SAVE_DIR,'CNN_woVEC'))
    print ("=======================--E-P-O-C-H--",epoch,"---========================================")

    with open (os.path.join(SAVE_DIR, 'info.log'), 'a') as f:
        f.write (str(datetime.datetime.now()))
        f.write("Epoch          "+str( epoch)+'\n')
        f.write('datacount      '+str(lendata)+'\n')
        f.write('dict           '+str(MAX_NB_WORDS)+'\n')
        f.write('EMBEDDING_DIM  '+str(EMBEDDING_DIM)+'\n')



