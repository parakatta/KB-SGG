# **Knowledge-based Scene Graph Generation in Medical Field**    

Scene understanding in the medical field based on visual context using object detection and representing it in the form of knowledge graphs in order to derive conclusions or understand contexts.
This problem statement highlights understanding the hospital environment and deciding the relationship between the objects detected in the image.  

![main_img](https://user-images.githubusercontent.com/83866928/235880897-bdd3b692-1e38-409f-9d15-5872eb9d331c.png)
The major contributions of our proposed work are:  
- A Knowledge- based representation in the form of scene graphs of images and scenarios seen in the medical field.
- Accurate analysis or understanding of the objects in the scene presented to it as an image.
- Publication of the accomplished work in domain-specific Conference/Journal.
- Contribution to future research in Scene understanding and Intent recognition.  

## Methodology

![arch_img](https://user-images.githubusercontent.com/83866928/235880903-00ba2e81-aa18-4624-8be8-93441562ff76.png)  

We used Faster RCNN model to train our custom dataset. Restricting our project to the medical field, our dataset consists of 11 classes. These classes depict the day to day basic scenarios in a hospital - cardiac_monitor, doctor, face_mask, iv_fluids, oxygen_mask, patient, stethoscope, syringe, ventilator, lightings, patient_bed.
The nodes and relationships are created using SVO triplets - Subject Verb and Objects. A complete triplet includes the predicate label and the class labels as well as the bounding box coordinates of the subject and object. All inferences will be based on the SVO triplets.
The predicate classes used are - USES, HAS, IN, LAY_ON, RECEIVES, CHECKS, READS, TREATS, INJECTED_TO, WEARS. The knowledge base is populated with the SVO triplets that depicts simple scenarios in the medical field. We have 12 nodes and 21 relationsips as of now.  
![kb_full](https://user-images.githubusercontent.com/83866928/235880886-9ba3caa8-faa5-43de-992d-8db298d44ba9.png)

## Performance  

We evaluate the object detection model using Precision, Recall, mAP for each bounding box detected via IoU (Intersection of Unions). Since we used bbox as the means of evaluation, Faster RCNN provides these metrics for each bbox involved.
The method shows the results of both object detection (bounding boxes and object labels) and edge predictions (relationship labels), i.e., a scene graph. Relationship labels are represented as directed edges.  

![Screenshot 2023-04-28 095228](https://user-images.githubusercontent.com/83866928/235881828-4bb59e92-b50e-4964-8540-ac10651e5b8e.png)
![map](https://user-images.githubusercontent.com/83866928/235882133-bf9d6b56-f146-4fbe-913d-7e5a1c91a275.png)
![train_loss_epoch](https://user-images.githubusercontent.com/83866928/235882139-c4044022-67a1-4b5c-aa19-632d5722e5f7.png)

## Future Work

Our project is an advancement to the current object detection technologies as we go one step further and provide the detection with some understanding behind the entities it detects. We make use of knowledge graphs to maintain and create a relationship between the objects detected.   
This problem statement highlights understanding the hospital environment and deciding the relationship between the objects detected in the image.  

Scene graph generation (SGG) is a semantic understanding task that goes beyond object detection and is closely linked to visual relationship detection. At present, scene graphs have shown their potential in different visionlanguage tasks such as image retrieval, image captioning, visual question answering (VQA) and image generation. The task of scene graph generation has also received sustained attention in the computer vision community.  
We hope to use this project to help the visually impaired and the elderly to better understand what’s in front of them. This is also a huge leap for research areas and robot training.


# IMPLEMENTATION

First, download the repo as zip file.
cd into your working directory.
#### 1. Install the dependencies.
``` pip install -r requirements.txt 
```
#### 2. Run the streamlit server
``` streamlit run main_deal.py 
```



