import os
import random
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import shutil as sh
import streamlit as st

# Set constants
img_h, img_w, num_channels = (380, 676, 3)

# Load and process the dataframe
df = pd.read_csv('C:/cogentinfo/yolo-learning/car-det/data/train_solution_bounding_boxes (1).csv')
df.rename(columns={'image': 'image_id'}, inplace=True)
df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])
df['x_center'] = (df['xmin'] + df['xmax']) / 2
df['y_center'] = (df['ymin'] + df['ymax']) / 2
df['w'] = df['xmax'] - df['xmin']
df['h'] = df['ymax'] - df['ymin']
df['classes'] = 0
df['x_center'] = df['x_center'] / img_w
df['w'] = df['w'] / img_w
df['y_center'] = df['y_center'] / img_h
df['h'] = df['h'] / img_h

# Display the dataframe

st.subheader('Random image before training:')

col3, col4 = st.columns(2)

# Random image display
index = list(set(df.image_id))
image = random.choice(index)
img_path = f'C:/cogentinfo/yolo-learning/car-det/data/training_images/vid_4_620.jpg'
img = cv2.imread(img_path)
st.write(f"Image ID 1: {image}")

with col3:
    st.image(img_path, caption='Random Image 1', use_column_width=True)




# Random image display
index = list(set(df.image_id))
image = random.choice(index)
img_path = f'C:/cogentinfo/yolo-learning/car-det/data/training_images/vid_4_2080.jpg'
img = cv2.imread(img_path)
st.write(f"Image ID 2: {image}")

with col4:
    st.image(img_path, caption='Random Image 2', use_column_width=True)




# Process and save the images and labels
source = 'training_images'
for fold in [0]:
    val_index = index[len(index) * fold // 5:len(index) * (fold + 1) // 5]
    for name, mini in tqdm(df.groupby('image_id')):
        if name in val_index:
            path2save = 'val2017/'
        else:
            path2save = 'train2017/'
        label_path = f'C:/cogentinfo/yolo-learning/car-det/processed_data/fold{fold}/labels/{path2save}'
        image_path = f'C:/cogentinfo/yolo-learning/car-det/processed_data/fold{fold}/images/{path2save}'
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        with open(f'{label_path}{name}.txt', 'w+') as f:
            row = mini[['classes', 'x_center', 'y_center', 'w', 'h']].astype(float).values
            row = row.astype(str)
            for j in range(len(row)):
                text = ' '.join(row[j])
                f.write(text + '\n')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        sh.copy(f"C:/cogentinfo/yolo-learning/car-det/data/{source}/{name}.jpg", f"{image_path}{name}.jpg")

# Display detected images
st.subheader('Some outputs after training')


col1, col2 = st.columns(2)

# Image 1
with col1:
    st.image('C:/cogentinfo/yolo-learning/yolov5/runs/detect/exp/vid_5_26660.jpg', caption='Detected Image 1', use_column_width=True)

# Image 2
with col1:
    st.image('C:/cogentinfo/yolo-learning/yolov5/runs/detect/exp/vid_5_400.jpg', caption='Detected Image 3', use_column_width=True)

# Image 3
with col1:
    st.image('C:/cogentinfo/yolo-learning/yolov5/runs/detect/exp/vid_5_26740.jpg', caption='Detected Image 5', use_column_width=True)

# Dummy placeholders for the remaining images in the grid
with col2:
    st.image('C:/cogentinfo/yolo-learning/yolov5/runs/detect/exp/vid_5_26640.jpg', caption='Detected Image 2', use_column_width=True)

with col2:
    st.image('C:/cogentinfo/yolo-learning/yolov5/runs/detect/exp/vid_5_27300.jpg', caption='Detected Image 4', use_column_width=True)

with col2:
    st.image('C:/cogentinfo/yolo-learning/yolov5/runs/detect/exp/vid_5_30920.jpg', caption='Detected Image 6', use_column_width=True)

