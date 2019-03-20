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
import os


def main():
    construct_vocab = False
    encode_images = False
    train = True

    # Read and Process Raw data
    data = CaptioningData()
    # Finding image files as data
    data.set_all_images(cfg.images_path)
    captions_dict = data.get_captions(cfg.token_file)
    caption_maxlen = data.get_caption_maxlen()

    # Construct vocabulary
    if construct_vocab:
        # get all caption to construct Vocab
        all_captions = data.get_all_captions()
        vocab = build_vocab(vocab_path=cfg.data_path, vocab_name=cfg.vocab_name, captions=all_captions, threshold=2)
    else:
        vocab = load_vocab(vocab_path=cfg.data_path, vocab_name=cfg.vocab_name)
    # print(vocab.word2idx)
    inception_encoding = Encoder()

    # train data
    if train:
        train_images = data.get_train_images(cfg.train_image_files)
        train_pairs = [ImgCaptionPair(img_id, captions_dict[img_id]) for img_id in train_images]

        # Image Encoding

        if encode_images:
            train_img_encoding = inception_encoding.encode_images(file_path=cfg.images_path, image_list=train_images,
                                                                  encoding_file=cfg.train_img_encoding_file)
        else:
            train_img_encoding = inception_encoding.load_image_encoding(encoding_file=cfg.train_img_encoding_file)

        train_data_generator = data_generator(vocab, train_pairs, train_img_encoding, batch_size=1800,
                                              max_len=caption_maxlen)
        # next(g)

    # Decoder model
    decoder = Decoder(vocab_size=len(vocab), embedding_size=300, input_shape=2048, caption_max_len=caption_maxlen)
    decoder_model = decoder.get_model()
    decoder_model.load_weights('best_weights.97-0.95.hdf5')

    if train:
        decoder_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        ckpt = ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0, save_best_only=False,
                               save_weights_only=False, mode='auto', period=30)
        best_ckpt = ModelCheckpoint('best_weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0,
                                    save_best_only=True, save_weights_only=False, mode='auto', period=1)
        decoder_model.fit_generator(train_data_generator, steps_per_epoch=30, epochs=100, callbacks=[ckpt, best_ckpt])

    decoder_model.save('decoder_model.h5')

    img_ids = data.get_val_images(cfg.val_image_files)
    img_name = img_ids[9]

    enc_img = inception_encoding.encode_single_img(file_path=cfg.images_path, img_name=img_name)

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
    #plt.show()

if __name__ == '__main__':
    #print(backend.tensorflow_backend._get_available_gpus())

    main()
