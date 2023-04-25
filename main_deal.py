import subprocess
import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import pandas as pd
import cv2
import torch
import glob as glob
from PIL import Image, ImageDraw
from py2neo import Graph
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
# from model import FasterRCNN


def get_scores():
    scores = []
    with open('./example_test_data/score.txt', 'r') as fi:
        for line in fi:
            scores.append(line.strip('\n'))
    fi.close()
    st.divider()
    st.write('### TEST PREDICTIONS COMPLETE')
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total FPS", value=scores[0][:5])
    col2.metric(label="Frame count", value=scores[1])
    col3.metric(label="Average FPS", value=scores[2][:5])
    st.divider()


def get_classes():
    clas = []
    with open('./example_test_data/clas.txt', 'r') as f:
        for line in f:
            clas.append(line.strip('\n'))
    f.close()
    return clas


def run_inference(image_name):
    OUT_DIR_NUM = 0
    with open('./example_test_data/dir_num.txt', 'r') as fil:
        OUT_DIR_NUM = fil.readline()
        print(OUT_DIR_NUM)
    fil.close()

    OUT_DIR_NUM = int(OUT_DIR_NUM)
    input_file = './example_test_data/test_image.jpg'
    weights_file = "./outputs/training/custom_training/last_model.pth"
    output_file = f"./outputs/inference/res_{OUT_DIR_NUM}/.jpg"

    command = f"python inference.py --input {input_file} --weights {weights_file}"

    # command = "python inference.py --input /content/drive/MyDrive/faster-rcnn-test/fastercnn-pytorch-training-pipeline/example_test_data/operating-room.jpg --weights outputs/training/custom_training/last_model_state.pth"

    subprocess.call(command.split())

    return output_file


# Define the Streamlit app
def main():
    # Set the title and description of the app
    tab1, tab2 = st.tabs(["About", "Demo"])
    with tab1:
        st.title("Knowledge-based Scene Graph Generation in Medical Field")
        im1 = Image.open('./main_img.png')
        st.image(im1, "Example of an image and it's scene graph")
        st.write('''
                    **Scene understanding** in the medical field  based on visual context using object detection and representing it in the form of knowledge graphs in order to derive conclusions or understand contexts.

    This problem statement highlights understanding the hospital environment and deciding the **relationship between the objects detected** in the image.

                    The major contributions of our proposed work are:
    - A Knowledge- based  representation in the form of scene graphs of images and scenarios seen in the medical field. 
    - Accurate analysis or understanding of the objects in the scene presented to it as an image. 
    - Publication of the accomplished work in domain-specific Conference/Journal. 
    - Contribution to future research in Scene understanding and Intent recognition.

    

    ''')
        st.divider()
        st.subheader("Methodology")
        #im4 = Image.open('G:/knowledge_graphs_faster_rcnn/block1.png')
        im4 = Image.open('./arch_img.png')
        st.image(im4, 'Architecture')
        st.write('''
                We used Faster RCNN model to train our custom dataset. Restricting our project to the medical field, our dataset consists of 11 classes.
                These classes depict the day to day basic scenarios in a hospital - *cardiac_monitor, doctor, face_mask, iv_fluids, oxygen_mask, patient, stethoscope, syringe, ventilator, lightings, patient_bed.* \n
                The nodes and relationships are created using SVO triplets - Subject Verb and Objects. A complete triplet includes the predicate label and the class labels as well as the bounding box coordinates of the subject and object. All inferences will be based on the SVO triplets.\n 
                The predicate classes used are - USES, HAS, IN, LAY_ON, RECEIVES, CHECKS, READS, TREATS, INJECTED_TO, WEARS. The knowledge base is populated with the SVO triplets that depicts simple scenarios in the medical field. We have 12 nodes and 21 relationsips as of now.
                ''')
        # Allow the user to upload an image file
        #im5 = Image.open('G:/knowledge_graphs_faster_rcnn/graph (1).png')
        im5 = Image.open('./kb_full.png')
        st.image(im5, 'Knowledge Base')

        st.divider()
        st.subheader('Performance')
        st.write('''
                We evaluate the object detection model using Precision, Recall, mAP for each bounding box detected via IoU (Intersection of Unions).
                Since we used bbox as the means of evaluation, Faster RCNN provides these metrics for each bbox involved. \n
                The method shows the results of both object detection (bounding boxes and object labels) and edge predictions (relationship labels), i.e., a scene graph.
                Relationship labels are represented as directed edges.
                ''')
        results_eval = pd.read_csv(
            './outputs/training/custom_training/results.csv')
        st.write(results_eval)
        imm = Image.open(
            './outputs/training/custom_training/map.png')
        st.image(imm, "mAP mean Average Precision")
        imm1 = Image.open(
            './outputs/training/custom_training/train_loss_epoch.png')
        st.image(imm1, "Training loss")
        st.subheader("Future Work")
        st.write('''
                 Scene graph generation (SGG) is a semantic
understanding task that goes beyond object detection and is
closely linked to visual relationship detection. At present,
scene graphs have shown their potential in different visionlanguage tasks such as image retrieval, image captioning, visual question answering (VQA) and image
generation. The task of scene graph generation has
also received sustained attention in the computer vision
community.\n
We believe that the biggest problem of classic scene graph generation (SGG) comes from noisy datasets. Classic scene graph generation datasets adopt a bounding box-based object grounding, which inevitably causes a number of issues:

- Coarse localization: bounding boxes cannot reach pixel-level accuracy,
- Inability to ground comprehensively: bounding boxes cannot ground backgrounds,
- Tendency to provide trivial information: current datasets usually capture frivolous objects like head to form trivial relations like person-has-head, due to too much freedom given during bounding box annotation.
- Duplicate groundings: the same object could be grounded by multiple separate bounding boxes.
                 ''')
    with tab2:
        st.write("Upload an image related to a scenario in a hospital")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])

        # If an image file is uploaded
        if uploaded_file is not None:
            # Open the image file using PIL
            st.image(uploaded_file)
            image = Image.open(uploaded_file)
            image = image.save(
                './example_test_data/test_image.jpg')

            # frame = np.array(image)
            # Run inference on the image and get the result
            result_image = run_inference(uploaded_file)
            res = Image.open(result_image)
            # Display the resulting image
            st.success("Object Detection complete!")
            st.image(res, caption='Detected objects', use_column_width=True)
            get_scores()
            clas = get_classes()
            graph_query(clas)


def inf_output():
    clas = []
    with open('./example_test_data/final_inf.txt', 'r') as f:
        for line in f:
            clas.append(line.strip('\n'))
    f.close()
    st.subheader('FINAL INFERENCE:')
    im2 = Image.open(
        './example_test_data/test_image.jpg')
    st.image(im2, "Input Image")
    for i in set(clas):
        st.write(f'##### {i}')
    # st.write(str(set(clas)))


def check_room(classes, clas, graph):
    final_inf = []
    op_room = ['doctor', 'iv_fluids', 'ventilator',
               'lightings', 'cardiac_monitor', 'patient_bed']
    if len(classes) < len(op_room):
        check = all(item in classes for item in op_room)
    else:
        check = all(item in op_room for item in classes)
    if check:
        # query3 = "MATCH (d)-[r1]->(n1:"+clas+") RETURN d,r1,n1"
        qu2 = "MATCH (d)-[r1]->(n1:"+clas+") WHERE n1.name ='" + \
            clas+"' AND NOT d.name = 'Icu_room' RETURN d, r1, n1"
        results3 = graph.run(qu2).data()
        if results3:
            st.write('### KNOWLEDGE BASED INFERENCE ON : ')
            st.subheader(f'Nodes labeled {clas}:')
            for row in results3:
                temp = ' '.join([row['d']['name'], row['r1']
                                ['name'], row['n1']['name']])
                final_inf.append(temp)
                st.write(row['d']['name'], row['r1']
                         ['name'], row['n1']['name'])
            with open('./example_test_data/final_inf.txt', 'a') as file:
                for pr in final_inf:
                    file.write(pr)
                    file.write('\n')
                file.close()
            st.divider()
        else:
            st.write(f'No relevant inference found for {clas}')
            st.divider()
        if clas == 'Syringe':
            clas = 'null'
    else:
        qu2 = "MATCH (d)-[r1]->(n1:"+clas+") WHERE n1.name ='" + \
            clas+"' AND NOT d.name = 'Operating_room' RETURN d, r1, n1"
        results3 = graph.run(qu2).data()
        if results3:
            st.write('### KNOWLEDGE BASED INFERENCE ON : ')
            st.subheader(f'Nodes labeled {clas}:')
            for row in results3:
                temp = ' '.join([row['d']['name'], row['r1']
                                ['name'], row['n1']['name']])
                final_inf.append(temp)
                st.write(row['d']['name'], row['r1']
                         ['name'], row['n1']['name'])
            with open('./example_test_data/final_inf.txt', 'a') as file:
                for pr in final_inf:
                    file.write(pr)
                    file.write('\n')
                file.close()
            st.divider()
        else:
            st.write(f'No relevant inference found for {clas}')
            st.divider()
        if clas == 'Syringe':
            clas = 'null'
    if clas == 'Syringe':
        qu5 = "MATCH (d:Syringe)-[r1]->(n1)  RETURN d, r1, n1"
        results6 = graph.run(qu5).data()
        if results6:
            st.write('### KNOWLEDGE BASED INFERENCE ON : ')
            st.subheader(f'Nodes labeled {clas}:')
            for row in results6:
                temp = ' '.join([row['d']['name'], row['r1']
                                ['name'], row['n1']['name']])
                final_inf.append(temp)
                st.write(row['d']['name'], row['r1']
                         ['name'], row['n1']['name'])
            with open('./example_test_data/final_inf.txt', 'a') as file:
                for pr in final_inf:
                    file.write(pr)
                    file.write('\n')
                file.close()
            st.divider()
        else:
            st.write(f'No relevant inference found for {clas}')
            st.divider()


def graph_query(classes):
    graph = Graph("neo4j://localhost:7687", auth=("neo4j", "password"))
    # Run a query to find all nodes labeled "Doctor"
    st.write("Predicted classes: ")
    st.write(classes)
    ans = ', '.join(classes)

    st.write(f'The image contains **{ans}**')
    st.divider()
    new_clas = classes.copy()
    for clas in classes:
        clas = clas.capitalize()
        tab_auto = st.tabs([clas])
        if clas == 'Doctor':
            # query1 = "MATCH (n:Node {label:'"+clas+"'}) RETURN n"
            if clas in new_clas:
                new_clas.remove(clas)

            for nw in new_clas:
                nw = nw.capitalize()
                query1 = "MATCH (d:Doctor)-[r1]->(n1:"+nw+") RETURN d, r1, n1"
                results1 = graph.run(query1).data()
                if results1:
                    st.write(" **Nodes labeled 'Doctor':**")
                    st.subheader(f'Nodes labeled {nw}:')
                    for row in results1:
                        # print(row['d'],row['r1'],row['n1'])
                        st.write(row['d']['name'], row['r1']
                                 ['name'], row['n1']['name'])
                    display_graph(results1)
            query1 = "MATCH (d:Doctor)-[r1]->(n1) RETURN d, r1, n1"
            results11 = graph.run(query1).data()
            if results11:
                st.write("### KNOWLEDGE BASED INFERENCES")
                for row in results11:
                    # print(row['d'],row['r1'],row['n1'])
                    st.write(row['d']['name'], row['r1']
                             ['name'], row['n1']['name'])
                display_graph(results11)

        elif clas == 'Patient':
            if clas in new_clas:
                new_clas.remove(clas)

            for nw in new_clas:
                nw = nw.capitalize()
                query1 = "MATCH (d:Patient)-[r1]->(n1:"+nw+") RETURN d, r1, n1"
                results1 = graph.run(query1).data()
                if results1:
                    st.subheader(f"Nodes labeled {nw}:")
                    for row in results1:
                        st.write(row['d']['name'], row['r1']
                                 ['name'], row['n1']['name'])
                    display_graph(results1)
            query2 = "MATCH (d:Patient)-[r1]->(n1) RETURN d, r1, n1"
            results2 = graph.run(query2).data()
            if results2:
                st.write("### KNOWLEDGE BASED INFERENCES")
                st.subheader("Nodes labeled 'Patients':")
                for row in results2:
                    st.write(row['d']['name'], row['r1']
                             ['name'], row['n1']['name'])
                display_graph(results2)
        else:
            query3 = "MATCH (d)-[r1]->(n1:"+clas+") RETURN d,r1,n1"
            results3 = graph.run(query3).data()
            if results3:
                st.subheader(f'Nodes labeled {clas} (object):')
                for row in results3:
                    st.write(row['d']['name'], row['r1']
                             ['name'], row['n1']['name'])
                display_graph(results3)
                check_room(classes, clas, graph)

            query4 = "MATCH (d:"+clas+")-[r1]->(n1) RETURN d,r1,n1"
            results4 = graph.run(query4).data()
            if results4:
                st.subheader(f'Nodes labeled {clas} (subject):')
                for row in results4:
                    st.write(row['d']['name'], row['r1']
                             ['name'], row['n1']['name'])
                display_graph(results4)
                check_room(classes, clas, graph)
    inf_output()


def display_graph(results):

    fig = plt.figure(figsize=(8, 6))
    G = nx.DiGraph()
    for record in results:
        G.add_edge(record['d']['name'], record['n1']
                   ['name'], label=record['r1']['name'])

    # draw graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(
        u, v): d['label'] for u, v, d in G.edges(data=True)})
    if G.edges:
        plt.show()
        st.pyplot(fig)


# Run the Streamlit app
if __name__ == '__main__':
    main()