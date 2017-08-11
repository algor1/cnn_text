
import os,sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.engine import Input, Model

from tensorflow.contrib.keras.python.keras.utils import to_categorical
from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.keras.python.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

TEXT_DATA_DIR = ".\\data"
GLOVE_DIR = "./glove.6B"

MAX_NB_WORDS=30000 #словарь
EMBEDDING_DIM=100
VALIDATION_SPLIT=0.05
NUM_ROWS_FROM_TEXT= 500
filename="training_text"
filename_v="training_variants"


texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids




# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 if sys.version_info < (3,):
#                     f = open(fpath)
#                 else:
#                     f = open(fpath, encoding='latin-1')
#                 t = f.read()
#                 i = t.find('\n\n')  # skip header
#                 if 0 < i:
#                     t = t[i:]
#                 texts.append(t)
#                 f.close()
#                 labels.append(label_id)
#
# print('Found %s texts.' % len(texts))


def read_data(filename,filename_v):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    labels_index['0'] = 0
    """Extract the first file enclosed in a zip file as a list of words"""
    diction=dict()
    ret=dict()
    i=0
    with open(filename_v, 'r', encoding="utf-8") as fv:
        for l in fv:
            data = tf.compat.as_str(l.strip()).split(',')
            diction[int(data[0])]=[data[1],data[2],int(data[3])]
            i += 1
#            print(i, diction,  data)
#            if i>5:
#                break
    i = 0
    with open(filename, 'r', encoding="utf-8") as f:

        for line in f:
            texts.append( line[line.find('||') + 2:])
            id = int(line[:line.find('||')])
            i+=1
            labels.append(diction[id][2])
            labels_index[str(diction[id][2])]=diction[id][2]
            #print(len(tf.compat.as_str(line.strip()).split()), diction[int(line[:line.find('||')])][2])
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
#exit(0)
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
print (x_train.shape)
print (x_val.shape)
print (y_train.shape)
print (y_val.shape)
# print (x_val,y_val)



#____________


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
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



embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#___________


# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             input_length=MAX_SEQUENCE_LENGTH)



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# happy learning!
#for j in range (3321,NUM_ROWS_FROM_TEXT):
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=5)
model.save('CNN_woVEC')