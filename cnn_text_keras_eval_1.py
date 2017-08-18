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
MAX_NB_WORDS_IN_TEXT=5000 # Если больше то режем на куски
EMBEDDING_DIM=100
VALIDATION_SPLIT=0.05
NUM_ROWS_FROM_TEXT= 6000
NUM_ROWS_SAVE_TO_TRAIN=100
NUM_ROWS_SAVE_TO_VAL=int(NUM_ROWS_SAVE_TO_TRAIN*VALIDATION_SPLIT)
NUM_EPOCHS=1
filename="./test_text"
filename_v="./test_variants"

tf.flags.DEFINE_string("loaddir", "Save1502895679", "Load data")


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

def predict(_t_w, _id,_word_index,_model,_sf,_model_shape1):
    sec = []
    sequences = []
    for w in _t_w:
        sec.append(_word_index.item().get(w, 0))
    sequences.append(sec)

    data = pad_sequences([sec], maxlen=_model_shape1)

    prediction = _model.predict(data, batch_size=1)


    return prediction[0]






with open(filename, 'r', encoding="utf-8") as f:
    with open(os.path.join(SAVE_DIR, 'submissionFile'), 'a') as sf:
        with open(os.path.join(SAVE_DIR, 'submissionFile_average'), 'a') as sfa:
            sf.write('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n')
            sfa.write('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n')
            for line in f:
                if i>0:
                    text= line[line.find('||') + 2:]
                    id =int(line[:line.find('||')])

                    t_w = text_to_word_sequence(text)
                    outputstr_m = str(id)
                    outputstr_a = str(id)
                    predict_list = []
                    # ----------
                    if len(t_w) > MAX_NB_WORDS_IN_TEXT:
                        for text_i in range(0, len(t_w), MAX_NB_WORDS_IN_TEXT):
                            if text_i + MAX_NB_WORDS_IN_TEXT - 1 < len(t_w):
                                t_w_splited=t_w[text_i:text_i + MAX_NB_WORDS_IN_TEXT - 1]
                                predict_list.append(predict(t_w_splited, id, word_index, model, sf,model_shape1))
                            elif len(t_w) - text_i > 50:
                                t_w_splited=t_w[text_i:len(t_w)]
                                predict_list.append(predict(t_w_splited, id, word_index, model, sf,model_shape1))
                    else:
                        predict_list.append(predict(t_w, id, word_index, model, sf,model_shape1))

                    # ----------------------
                    max_predict=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                    for prediction in predict_list:

                        for j, p_i in enumerate(prediction):
                            if j > 0 and p_i>max_predict[j-1]:
                                max_predict[j-1]=p_i

                    for m_p in max_predict:
                        outputstr_m += "," + "%.2f" % m_p

                    print(outputstr_m)
                    sf.write(outputstr_m + '\n')
                    #------------------------
                    sum_predict = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    for prediction in predict_list:

                        for j, p_i in enumerate(prediction):
                            if j > 0:
                                sum_predict[j - 1] += p_i

                    for m_p in max_predict:
                        outputstr_a += "," + "%.2f" % (m_p/len(predict_list))

                    print(outputstr_a)
                    sfa.write(outputstr_a + '\n')

                i+=1
                if i>=NUM_ROWS_FROM_TEXT :
                    break

print("saved in "+ os.path.join(SAVE_DIR, 'submissionFile'))

    # with open(filename, 'r', encoding="utf-8") as f:
    #     for line in f:
    #
    #         if i > 0 and len(line) > 100:
    #
    #             id = int(line[:line.find('||')])
    #             t_w = text_to_word_sequence(line[line.find('||') + 2:])
    #             if len(t_w) > MAX_NB_WORDS_IN_TEXT:
    #                 for text_i in range(0, len(t_w), MAX_NB_WORDS_IN_TEXT):
    #                     if text_i + MAX_NB_WORDS_IN_TEXT - 1 < len(t_w):
    #                         prerdict()   texts.append(' '.join(t_w[text_i:text_i + MAX_NB_WORDS_IN_TEXT - 1]))
    #                         labels.append(diction[id][2])
    #                     elif len(t_w) - text_i > 50:
    #                         prerdict()  texts.append(' '.join(t_w[text_i:len(t_w)]))
    #                         labels.append(diction[id][2])
    #             else:
    #                 prerdict() texts.append(' '.join(t_w))
    #                 labels.append(diction[id][2])
    #
    #             labels_index[str(diction[id][2])] = diction[id][2]
    #         i += 1
    #         if i >= NUM_ROWS_FROM_TEXT:
    #             break