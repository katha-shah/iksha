import os
from data_loader import CaptioningData, ImgCaptionPair, data_generator
from vocab import build_vocab, load_vocab
from config import Config as cfg
from model import Encoder, Decoder
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras import backend
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt


def beam_search_predictions(vocab, enc_img, decoding_model, caption_maxlen, beam_index=3):
    start = [vocab("<start>")]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < caption_maxlen:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=caption_maxlen, padding='post')
            #enc_img = encoding_model.encode_single_img(file_path=cfg.images_path, img_name=img_name)
            preds = decoding_model.predict([np.array([enc_img]), np.array(par_caps)])
            word_pred_debug = vocab.idx2word[np.argmax(preds[0])]
            # print(len(preds), word_pred_debug)


            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again

            word_preds = np.argsort(preds[0])[-beam_index:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp

        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [vocab.idx2word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


def main():
    infer_img_path = "C:/Users/Crypto/PycharmProjects/segmented_style_transfer/data"
    infer_img_name = 'two_boys.jpg'
    # Read and Process Raw data
    data = CaptioningData()
    # Finding image files as data
    data.set_all_images(cfg.images_path)
    captions_dict = data.get_captions(cfg.token_file)
    caption_maxlen = data.get_caption_maxlen()

    vocab = load_vocab(vocab_path=cfg.data_path, vocab_name=cfg.vocab_name)
    # print(vocab.word2idx)
    inception_encoding = Encoder()

    # Decoder model
    decoder = Decoder(vocab_size=len(vocab), embedding_size=300, input_shape=2048, caption_max_len=caption_maxlen)
    decoder_model = decoder.get_model()
    decoder_model.load_weights('best_weights.97-0.95.hdf5')

    img_ids = data.get_val_images(cfg.val_image_files)
    img_name = img_ids[19]

    enc_img = inception_encoding.encode_single_img(file_path=cfg.images_path, img_name=img_name)
    # enc_img = inception_encoding.encode_single_img(file_path=infer_img_path, img_name=infer_img_name)

    caption = ["<start>"]
    while True:
        par_caps = [vocab(i) for i in caption]
        par_caps = sequence.pad_sequences([par_caps], maxlen=40, padding='post')
        preds = decoder_model.predict([np.array([enc_img]), np.array(par_caps)])
        word_pred = vocab.idx2word[np.argmax(preds[0])]
        caption.append(word_pred)

        if word_pred == "<end>" or len(caption) > 40:
            break

    full_img_path = os.path.join(cfg.images_path, img_name)
    print(captions_dict[img_name])
    print(full_img_path)
    print(' '.join(caption[1:-1]))

    caption_3 = beam_search_predictions(vocab, enc_img, decoder_model, caption_maxlen,
                                        beam_index=2)
    caption_5 = beam_search_predictions(vocab, enc_img, decoder_model, caption_maxlen,
                                        beam_index=5)
    caption_7 = beam_search_predictions(vocab, enc_img, decoder_model, caption_maxlen,
                                        beam_index=7)

    print("3", caption_3)
    print("5", caption_5)
    print("7", caption_7)
    # plt.show()


if __name__ == '__main__':
    main()
