# iksha
Describes what it sees </br>
#### Introduction </br>
*An application designed for specially abled people having vision
impairment. These people need assistance to look around the things in the world.
Many such conventional solutions are available to help these people. But with this
project and the help of A.I., we are designing description model which entails the
surrounding around us. This description is given in simple plain English sentence.</br></br>
We use machine learning algorithms to solve this problem to greater accuracy.
We divide this solution into two parts (i) Feature Extraction (ii) Language Model.
For feature extraction we use neural networks to get the feature vector of images and
processing the data. Also we use tranfer learning to use the pre-trained model over
our input sets. Language model will make use of NLP concepts to generate the
meaningful sentences in plain English language.*

#### [Dataset](https://forms.illinois.edu/sec/229675)
* FLICKR30k - Contains 30,000 images with its caption in English language splitting 1000 images for validation and 1000 images for Testing

#### Architecture
*Encoder* </br></br>
  • CNN can be thought of as an Encoder. </br>
  •  CNN is a widely used image feature extraction technique for object detection and image classification. Transfer learning is used to obtain the features of the images from       the       dataset </br>

*Decoder* </br></br>
• Decoder is the LSTM </br>
• Language modelling is done at the word level. </br>
• The first time step receives the encoded output from the encoder and also the <START> vector.
  
  





