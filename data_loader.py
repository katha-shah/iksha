from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from config import Config as cfg
import random


class ImgCaptionPair(object):

    def __init__(self, img_id, captions):
        self._img_id = img_id
        # print(captions)
        random.shuffle(captions)
        self._captions = captions
        self.curr_idx = 0

    def __call__(self):
        # print("Returning img caption")
        # print(self._img_id)
        if self._captions:
            curr_caption = '<start> ' + self._captions[self.curr_idx] + ' <end>'
            self.curr_idx = (self.curr_idx + 1) % len(self._captions)
            return self._img_id, curr_caption.lower()


class CaptioningData(object):

    def __init__(self):
        self.captions_dict = {}
        self.all_images = []
        self.all_captions = []
        self.max_len_caption = -1
        self.train_images = []
        self.val_images = []
        self.test_images = []

    def get_captions(self, token_file):
        with open(token_file, 'r') as f:
            captions = f.read().strip().split('\n')

        for row in captions:
            row = row.split('\t')
            row[0] = row[0][:len(row[0]) - 2]
            if row[0] in self.captions_dict:
                self.captions_dict[row[0]].append(row[1])
            else:
                self.captions_dict[row[0]] = [row[1]]

        self.all_captions = [caption for captions in self.captions_dict.values() for caption in captions]
        self.max_len_caption = max([len(caption.split()) for caption in self.all_captions])
        # caption_dict[img_name] = list of captions
        return self.captions_dict

    def get_all_captions(self):
        if not self.captions_dict:
            raise ValueError('Use [get_captions] method to find captioning mapping')
        return self.all_captions

    def set_all_images(self, image_dir):
        """
        searching all image files in image dir
        """
        self.all_images = [img[len(image_dir):] for img in glob(image_dir + '*jpg')]
        # print(self.all_images)
        return self.all_images

    def _process_data_names(self, images_file):
        """
        Helper fn to read and validate train, validation, and test image names
        """
        if not self.all_images:
            raise ValueError('Use [set_all_images] method to find valid set of images')
        with open(images_file, 'r') as f:
            images = f.read().strip().split('\n')
            images = set(images)
            # keep the image name only if it is resent in all_images list
            valid_images = [image for image in images if image in self.all_images]

        return valid_images

    def get_train_images(self, train_img_file):
        self.train_images = self._process_data_names(train_img_file)
        return self.train_images

    def get_val_images(self, val_img_file):
        self.val_images = self._process_data_names(val_img_file)
        return self.val_images

    def get_test_images(self, test_img_file):
        self.test_images = self._process_data_names(test_img_file)
        return self.test_images

    def print_data_split(self):
        print("Train:{} Validation:{} Test:{}".format(len(self.train_images), len(self.val_images),
                                                      len(self.test_images)))

    def get_caption_maxlen(self):
        if self.max_len_caption == -1:
            raise ValueError('Use [get_captions] method to find captioning mapping')
        return self.max_len_caption + 2  # Since <start> and <end> tags will be appended


def data_generator(vocab, ic_pairs, img_encoding, batch_size=32, max_len=20):
    """
    :param vocab: the tokens of all the caption data
    :param ic_pairs: img_id & caption
    :param img_encoding: CNN encoding dict of a img_id
    :param batch_size
    :param max_len: max length of
    :return: [[encoded_img, partial_caps], next_words]

    eg:
    img1 : 511 34 52 123 7 9 512
    img2 : 511 2 65 512
    maxlen = 8
            |
            |
        Data generated
            |
    a<>  : x<>                            : y<>
    img1 : 511  0   0   0   0   0   0   0 : 34
    img1 : 511  34  0   0   0   0   0   0 : 52
    img1 : 511  34  52  0   0   0   0   0 : 123
    ...
    img2 : 511  2   65  512 0   0   0   0 : 512

    """
    partial_caps = []
    next_words = []
    images = []

    vocab_size = len(vocab)
    count = 0

    while True:
        for ic_pair in ic_pairs[:]:
            current_image, current_caption = ic_pair()
            current_image = img_encoding[current_image]

            for i in range(len(current_caption.split()) - 1):
                count += 1
                words = current_caption.split()
                partial = [vocab(word) for word in words[:i + 1]]
                partial_caps.append(partial)

                # Initializing with zeros to create a one-hot encoding matrix
                # This is what we have to predict
                # Hence initializing it with vocab_size length
                n = np.zeros(vocab_size)
                next_word = words[i + 1]
                # Setting the next word to 1 in the one-hot encoded matrix
                n[vocab(next_word)] = 1
                next_words.append(n)

                images.append(current_image)
                if count >= batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                    # print("-D-:: Shapes: images={}, partial_caps={}, next_words={}".format(images.shape,
                    #                                                                        partial_caps.shape,
                    #                                                                        next_words.shape))
                    yield [[images, partial_caps], next_words]
                    partial_caps = []
                    next_words = []
                    images = []
                    count = 0
