from config import Config as cfg
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Add, Concatenate, Activation, \
    Flatten
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import image
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from tqdm import tqdm
import os
import pickle


class Encoder(object):

    def __init__(self):
        self.cnn_model_name = "InceptionV3"
        model = InceptionV3(weights='imagenet')
        input_layer = model.input
        hidden_layer = model.layers[-2].output
        self.my_model = Model(input_layer, hidden_layer)

    def _encode_util(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # pre-process image
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def _encode_images(self, image_path, image_list):
        encoded_images = {}
        for img in tqdm(image_list):
            in_image = self._encode_util(os.path.join(image_path, img))
            out_image = self.my_model.predict(in_image)
            encoded_images[img] = out_image.reshape(out_image.shape[1])
        return encoded_images

    def _save_encoded_img(self, encoding_dict, encoding_file):
        print("Saving encoding data: {}...".format(encoding_file))
        with open(encoding_file, 'wb') as f:
            pickle.dump(encoding_dict, f)

    def encode_images(self, file_path, image_list, encoding_file):
        encoded_images = self._encode_images(file_path, image_list)
        self._save_encoded_img(encoded_images, encoding_file)
        return encoded_images

    def load_image_encoding(self, encoding_file):
        print("Loading encoding data: {}...".format(encoding_file))
        with open(encoding_file, 'rb') as f:
            return pickle.load(f)

    def encode_single_img(self, file_path, img_name):
        in_img = self._encode_util(os.path.join(file_path, img_name))
        encoded_img = self.my_model.predict(in_img)
        return encoded_img.reshape(encoded_img.shape[1])



class Decoder(object):

    def __init__(self, vocab_size=5000, embedding_size=300, input_shape=2048, caption_max_len=30):
        image_model = Sequential([
            Dense(embedding_size, input_shape=(input_shape,), activation='relu'),
            RepeatVector(caption_max_len)
        ])
        caption_model = Sequential([
            Embedding(vocab_size, embedding_size, input_length=caption_max_len),
            LSTM(256, return_sequences=True),
            TimeDistributed(Dense(embedding_size))
        ])

        # FIXME: Refactor final_model name to a different name [decoder_model]
        self.final_model = Sequential([
            Merge([image_model, caption_model], mode='concat', concat_axis=1),
            #Bidirectional(LSTM(256, return_sequences=True)),
            Bidirectional(LSTM(256, return_sequences=False)),
            Dense(vocab_size),
            Activation('softmax')
        ])

    def get_model(self):
        return self.final_model

    def model_summary(self):
        print(self.final_model.summary())
