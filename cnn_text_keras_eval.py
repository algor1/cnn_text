import datetime
import os,sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.models import load_model

from tensorflow.contrib.keras.python.keras.utils import to_categorical
from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.keras.python.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
d_now=datetime.datetime.now()
TEXT_DATA_DIR = "./data"
GLOVE_DIR = "./glove.6B"
SAVE_DIR= "./Save"+ str(int(d_now.timestamp()))
if not os.path.exists(SAVE_DIR): os.mkdir(SAVE_DIR)

MAX_NB_WORDS=180000 #словарь
EMBEDDING_DIM=100
VALIDATION_SPLIT=0.05
NUM_ROWS_FROM_TEXT= 6000
NUM_ROWS_SAVE_TO_TRAIN=100
NUM_ROWS_SAVE_TO_VAL=int(NUM_ROWS_SAVE_TO_TRAIN*VALIDATION_SPLIT)
NUM_EPOCHS=1
filename="./test_text"
filename_v="./test_variants"

tf.flags.DEFINE_string("loaddir", "Save1502701313", "Load data")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

word_index = np.load (os.path.join(FLAGS.loaddir, 'word_index.npy'))
model= load_model(os.path.join(FLAGS.loaddir, 'CNN_woVEC'))
model_shape1=model.input_shape[1]
i = 0
with open(filename, 'r', encoding="utf-8") as f:
    with open(os.path.join(SAVE_DIR, 'submissionFile'), 'a') as sf:
        sf.write('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n')
        for line in f:
            if i>0:
                text= line[line.find('||') + 2:]
                id =int(line[:line.find('||')])

                t_w = text_to_word_sequence(text)
                sec = []
                sequences = []
                for w in t_w:
                    sec.append(word_index.item().get(w, 0))
                sequences.append(sec)

                data = pad_sequences([sec], maxlen=model_shape1)

                prediction = model.predict(data, batch_size=1)
                outputstr = str(id)
                j = 0
                for p_i in prediction[0]:
                    if j > 0:
                        outputstr += "," + "%.2f" % p_i
                    j += 1
                print(outputstr)
                sf.write(outputstr + '\n')
            i+=1
            if i>=NUM_ROWS_FROM_TEXT :
                break

print("saved in "+ os.path.join(SAVE_DIR, 'submissionFile'))
#
#
#     sequences =[]
#
#
#     t_w=text_to_word_sequence(t)
#     sec=[]
#     for w in t_w:
#         sec.append( word_index.item().get(w,0))
#
#     sequences.append(sec)
#
# data = pad_sequences(sequences,maxlen=model.input_shape[1])
#
#
# prediction= model.predict(data,batch_size=10)
#
# with open(os.path.join(SAVE_DIR,'submissionFile'),'a') as f:
#     f.write('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n')
#     for i in range (len(prediction)):
#         outputstr=str(id_text[i])
#         j=0
#         for p_i in prediction[i]:
#             if j>0:
#                 outputstr+=","+"%.2f"%p_i
#             j+=1
#         print (outputstr)
#         f.write(outputstr+'\n')










