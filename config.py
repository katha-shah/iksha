import os

class Config(object):
	data_path = 'D:/padh.ai/data/Flickr8k'
	text_path = os.path.join(data_path, 'textdata')
	token_file = os.path.join(text_path, 'Flickr8k.token.txt')
	images_path = os.path.join(data_path, 'images/')
	train_image_files = os.path.join(text_path, 'Flickr_8k.trainImages.txt')
	val_image_files = os.path.join(text_path, 'Flickr_8k.devImages.txt')
	test_image_files = os.path.join(text_path, 'Flickr_8k.testImages.txt')

	# pickle files
	vocab_name = "vocab.pkl"
	train_img_encoding_file = os.path.join(data_path, "train_encod.pkl")
