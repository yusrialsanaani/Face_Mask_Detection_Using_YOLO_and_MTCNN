# Face_Mask_Detection_Using_YOLO_and_MTCNN


Masks play a crucial role in protecting the health of individuals against respiratory diseases. So, a face mask detection system can be implemented to automate the detection and notify the individuals, hence protecting their health. The goal of this project is to use deep learning (DL), which has shown excellent results in many real-life applications, to ensure accurate facemask detection pipeline. Two face mask detection pipelines have been implemented in this project. 

The first pipeline is a two-stage detector consists of convolutional neural network (CNN) classifier and Multi-Task Cascaded Convolutional Neural Networks (MTCNN) detector. The methodology used in this part is as shown in the figure below:

![image](https://user-images.githubusercontent.com/89004966/163080746-57a20ec9-081e-4a96-8686-0b193f50f2ba.png)

In the first pipeline, the CNN classifier has been tuned by using transfer learning approach based on VGG19 model

The second pipeline is a one-stage detector using YOLO such that YOLO network performs both object detection and class classification in one stage without a need to external classifier. The methodology used in this part is as shown in the figure below:

![image](https://user-images.githubusercontent.com/89004966/163080890-491646b8-b1cc-479f-a5e0-2afcb374471a.png)



All models in both pipelines have been trained, validated, and evaluated using different metrics such as accuracy, recall, precision, confusion matrix, ROC curve. Both pipelines perform well, however YOLO model performs better in term of detection accuracy and speed compared with MTCNN pipeline.
