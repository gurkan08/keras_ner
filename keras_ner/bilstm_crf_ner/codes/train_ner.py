
from keras_contrib.layers import CRF
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import pickle

class Params(object):
    epochs = 50
    batch_size = 64
    lr = 0.0005
    validation_split = 0.1
    max_sent_size = 47
    embed_size = 100
    lstm_units = 50

class Main(object):

    word2id = {}
    tag2id = {}
    sentence_tokenizer = None
    tag_tokenizer = None

    def __init__(self):
        pass

    @staticmethod
    def read_dataset(dataset_dir):
        sentence_list = []
        label_list = []
        with open(dataset_dir) as f:
            lines = f.readlines()
            sentence = ""
            label = ""
            for line in lines:
                line = line.strip()
                if line != "":
                    if "	" in line:
                        words = line.split("	")
                    elif " " in line:
                        words = line.split(" ")
                    sentence += words[1].strip() + " "
                    label += words[0].strip() + " "
                else:
                    sentence_list.append(sentence.lower().strip())
                    label_list.append(label.lower().strip())
                    sentence = ""
                    label = ""
        return sentence_list, label_list

    @staticmethod
    def preproces(train_sentence_list, train_tag_list, test_sentence_list, test_tag_list):
        """
        http://buyukveri.firat.edu.tr/2018/06/04/metin-on-isleme-adimlari-icin-keras-tokenizer-sinifi-kullanimi/
        """
        sentence_list = train_sentence_list + test_sentence_list
        Main.sentence_tokenizer = Tokenizer(oov_token="UNK") # 0 index reserved as padding_value
        Main.sentence_tokenizer.fit_on_texts(sentence_list)
        # print("--> ", Main.sentence_tokenizer.word_index)
        # print("--> ", Main.sentence_tokenizer.word_counts["moviesfsdf"])
        # print("--> ", Main.sentence_tokenizer.word_index["moviedfsf"])
        sentence_list = Main.sentence_tokenizer.texts_to_sequences(sentence_list)
        sentence_list = pad_sequences(sentence_list, maxlen=Params.max_sent_size, padding="post", value=0.)
        # save tokenizer
        with open("../model/sentence_tokenizer.pickle", 'wb') as handle:
            pickle.dump(Main.sentence_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tag_list = train_tag_list + test_tag_list
        # "-" ve "_" yi sildik çünkü "b-RATINGS_AVERAGE" ayrı ayrı kelime olarak bölüyor !
        Main.tag_tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
        Main.tag_tokenizer.fit_on_texts(tag_list)
        tag_list = Main.tag_tokenizer.texts_to_sequences(tag_list)
        tag_list = pad_sequences(tag_list, maxlen=Params.max_sent_size, padding="post", value=Main.tag_tokenizer.word_index["o"]) # "o" ile padding yap
        # error : IndexError: index 26 is out of bounds for axis 1 with size 26 ----> i - 1  ekledik
        tag_list = [to_categorical(i - 1, num_classes=len(Main.tag_tokenizer.word_index)) for i in tag_list]
        # save tokenizer
        with open("../model/tag_tokenizer.pickle", 'wb') as handle:
            pickle.dump(Main.tag_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return sentence_list[:len(train_sentence_list)], \
               tag_list[:len(train_sentence_list)], \
               sentence_list[len(train_sentence_list):], \
               tag_list[len(train_sentence_list):]

    @staticmethod
    def run_train(train_data, train_label):
        model_obj = MyModel(max_sentence_size=Params.max_sent_size,
                            embed_size=Params.embed_size,
                            vocab_size=len(Main.sentence_tokenizer.word_index),
                            lstm_units=Params.lstm_units,
                            tag_size=len(Main.tag_tokenizer.word_index))
        model = model_obj.get_model()
        adam = Adam(lr=Params.lr, beta_1=0.9, beta_2=0.999)
        # error link: https://github.com/keras-team/keras/issues/11749
        # model.compile(optimizer=adam, loss=model_obj.get_crf_loss(), metrics=[model_obj.get_crf_acc(), "accuracy"]) # error: tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [30] vs. [30,47]
        model.compile(optimizer=adam, loss=model_obj.get_crf_loss(), metrics=
            [model_obj.get_crf_acc()]) # metrics=[model_obj.get_crf_acc(), "accuracy"] # "accuracy yi çıkardık hata çözüldü !"
        print("------------model summary-------------")
        print(model.summary())
        history = model.fit(np.array(train_data),
                            np.array(train_label),
                            batch_size=Params.batch_size,
                            epochs=Params.epochs,
                            validation_split=Params.validation_split,
                            verbose=1)
        model.save("../model/model_" + str(Params.epochs) + ".h5")

class MyModel(object):

    def __init__(self, max_sentence_size, embed_size, vocab_size, lstm_units, tag_size):
        input_layer = Input(shape=(max_sentence_size,))
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embed_size, mask_zero=True) \
            (input_layer) # mask_zero=True because we used zero_padding
        bilstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(embed_layer)
        # bilstm_layer = LSTM(lstm_units, return_sequences=True)(embed_layer) # optional
        timedist_layer = TimeDistributed(Dense(tag_size, activation="relu"))(bilstm_layer)
        self.crf = CRF(tag_size)
        crf_layer = self.crf(timedist_layer)
        self.model = Model(input_layer, crf_layer)

    def get_model(self):
        return self.model

    def get_crf_loss(self):
        return self.crf.loss_function

    def get_crf_acc(self):
        return self.crf.accuracy

if __name__ == '__main__':

    train_dataset_dir = "../data/engtrain.bio"
    test_dataset_dir = "../data/engtest.bio"
    train_data, train_label = Main.read_dataset(train_dataset_dir)
    test_data, test_label = Main.read_dataset(test_dataset_dir)
    #print(len(train_data))
    #print(train_data[0])
    #print(len(train_label))
    #print(train_label[0])

    train_data, train_label, test_data, test_label = Main.preproces(train_data, train_label, test_data, test_label)
    #print(len(train_data))
    #print(train_data[6])
    #print(len(train_label))
    #print(train_label[6])

    Main.run_train(train_data, train_label)

