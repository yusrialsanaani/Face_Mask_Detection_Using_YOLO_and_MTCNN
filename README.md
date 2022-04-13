# Face_Mask_Detection_Using_YOLO_and_MTCNN


Masks play a crucial role in protecting the health of individuals against respiratory diseases. So, a face mask detection system can be implemented to automate the detection and notify the individuals, hence protecting their health. The goal of this project is to use deep learning (DL), which has shown excellent results in many real-life applications, to ensure accurate facemask detection pipeline. Two face mask detection pipelines have been implemented in this project. 

# Face Mask Detection Pipeline Using MTCNN and CNN Classifier

The first pipeline is a two-stage detector consists of convolutional neural network (CNN) classifier and Multi-Task Cascaded Convolutional Neural Networks (MTCNN) detector. The methodology used in this part is as shown in the figure below:

![image](https://user-images.githubusercontent.com/89004966/163080746-57a20ec9-081e-4a96-8686-0b193f50f2ba.png)

**The datasets used in this part is Face Mask Detection ~12K Images from Kaggle:
https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset

The dataset distribution over classes is shown in the figure below:

![image](https://user-images.githubusercontent.com/89004966/163081325-b879dafc-8737-4405-b56f-e61051641c47.png)

## CNN Classifier Training, validation, and Evaluation

### The Loss and Accuracy curves for CNN Model are shown below:

![image](https://user-images.githubusercontent.com/89004966/163081474-22a218e0-80b1-4dad-a4c0-0c6bb8652c3e.png)

### The Classification Metrics for CNN Model are shown below:

![image](https://user-images.githubusercontent.com/89004966/163081707-02613987-aea2-4d2e-b006-05c54d19674d.png)

### The ROC curve for CNN Model are shown below:

![image](https://user-images.githubusercontent.com/89004966/163081849-1ea7d061-a536-4ee9-84d1-1a4e73954110.png)

## CNN Classifier with Transfer Learning (VGG19 Model)

In the first pipeline, the CNN classifier has been tuned by using transfer learning approach based on VGG19 model.

### The Loss and Accuracy curves for VGG Model are shown below:

![image](https://user-images.githubusercontent.com/89004966/163082181-0a3d4597-c416-4c0e-9cf6-4bd277bcdccc.png)

### The Classification Metrics for VGG Model are shown below:

![image](https://user-images.githubusercontent.com/89004966/163082225-20c58720-2d0e-4a84-9ddf-3b513f164f5a.png)

### The ROC curve for VGG Model are shown below:

![image](https://user-images.githubusercontent.com/89004966/163082254-8474711c-a029-47c9-96e0-db6469c021c5.png)

### A quick comparison between CNN model and the tuned CNN model using transfer learning (VGG model) in term of metrics is show below:

![image](https://user-images.githubusercontent.com/89004966/163082651-67a0df7a-c6f3-45c3-9d39-99ead89fecf9.png)


** The CNN model with transfer learning performs better than the basic CNN model, so it will be used with MTCNN to create two-stage face mask detector as shown in the diagram below:

![image](https://user-images.githubusercontent.com/89004966/163082487-9f050744-10ee-47a4-95fc-7f371a109cdd.png)

** Some Outputs using MTCNN pipeline are shown below:**
![MTCNN_Outputs](https://user-images.githubusercontent.com/89004966/163082747-0a6b864e-43c9-4c58-b04a-74734677dabf.gif)



The second pipeline is a one-stage detector using YOLO such that YOLO network performs both object detection and class classification in one stage without a need to external classifier. The methodology used in this part is as shown in the figure below:

![image](https://user-images.githubusercontent.com/89004966/163080890-491646b8-b1cc-479f-a5e0-2afcb374471a.png)



All models in both pipelines have been trained, validated, and evaluated using different metrics such as accuracy, recall, precision, confusion matrix, ROC curve. Both pipelines perform well, however YOLO model performs better in term of detection accuracy and speed compared with MTCNN pipeline.
