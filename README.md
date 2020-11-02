# Iksha - Surrounding Description Model for Visually Impaired
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

### Dataset*
• [FLICKR30k](http://shannon.cs.illinois.edu/DenotationGraph/)
 - Contains 30,000 images with its caption in English language splitting 1000 images for validation and 1000 images for Testing

### Architecture
*Encoder* </br></br>
  • CNN can be thought of as an Encoder. </br>
  •  CNN is a widely used image feature extraction technique for object detection and image classification. Transfer learning is used to obtain the features of the images from       the       dataset (Inception v3)</br>

*Decoder* </br></br>
• Decoder is the Bi-directional Deep LSTM </br>
• Language modelling is done at the word level. </br>
• The first time step receives the encoded output from the encoder and also the <START> vector.
  
 ### Results
 
 Upon running the model for 70 epochs, we’ve attained BLEU score ~ 0.56 which is pretty good given the limited training dataset and computation power. </br>
• **Describes without error** </br>
  ![image](https://user-images.githubusercontent.com/24832637/97819663-e76b3b00-1c5e-11eb-820f-b4484a31102a.png)

• **Describes with minor errors** </br>
![image](https://user-images.githubusercontent.com/24832637/97820026-ff43be80-1c60-11eb-9cf9-d328125d7def.png)




  
  
  
  





