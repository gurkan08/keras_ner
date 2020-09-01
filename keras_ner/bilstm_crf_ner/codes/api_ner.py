
from keras_contrib.layers import CRF
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import tensorflow as tf

graph = tf.get_default_graph() # gerekli !, https://github.com/tensorflow/tensorflow/issues/14356

model_dir = "keras_ner/bilstm_crf_ner/model"
max_sent_size = 47
n_tag = 26
crf = CRF(n_tag)
with open(os.path.join(model_dir, "sentence_tokenizer.pickle"), "rb") as handle:
    sentence_tokenizer = pickle.load(handle)
with open(os.path.join(model_dir, "tag_tokenizer.pickle"), "rb") as handle:
    tag_tokenizer = pickle.load(handle)
model = load_model(os.path.join(model_dir, "model_50.h5"), custom_objects={"CRF": CRF,
                                                                           "crf_loss": crf.loss_function,
                                                                           "crf_viterbi_accuracy": crf.viterbi_acc})
print("----loaded model summary----------")
print(model.summary())

def _api(text):
    text = text.rstrip().lower()
    out = sentence_tokenizer.texts_to_sequences([text])  # list [] format
    # print(out)
    out = pad_sequences(out, maxlen=max_sent_size, padding="post", value=0.)
    # print(out)

    with graph.as_default(): # gerekli !
        pred = model.predict(out, verbose=1)
    # print(pred)
    # print(pred.shape) # (1, 47, 26)

    result = []
    sent_size = len(text.split())
    for id in range(sent_size):
        _index = list(pred[0, id, :]).index(1.0)
        # print(_index)
        for key, value in tag_tokenizer.word_index.items():
            if value == (_index + 1): # + 1
                result.append(key)
    return " ".join(result)

"""
# test samples
text = "show me films with drew barrymore from the 1980s"
text = "who directed the film pulp fiction that starred john travolta"
text = "which film has the highest viewer rating this year"
text = "what was the first movie in color"
text = "who diected the first james bond movies"
text = "list childrens movies with billy crystal"
text = "gürkan şahin ytü bilgisayar mühendisliği"
"""

