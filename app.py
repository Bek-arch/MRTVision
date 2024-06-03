import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
from util import visualize, set_background

set_background('./5c6dc1b5b81bcfd4c5da9b3948c2e879-1.jpg')

st.title('Обнаружение опухолей на МРТ мозга')
st.header('Пожалуйста, загрузите изображение')

uploaded_file = st.file_uploader('Загрузите изображение в формате PNG, JPG или JPEG', type=['png', 'jpg', 'jpeg'])

config = get_cfg()
config_path = model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml')
config.merge_from_file(config_path)
config.MODEL.WEIGHTS = './model/model.pth'
config.MODEL.DEVICE = 'cpu'  

predictor = DefaultPredictor(config)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    prediction_outputs = predictor(image_np)

    detection_threshold = 0.5
    detected_boxes = []
    for index, box in enumerate(prediction_outputs["instances"].pred_boxes):
        if prediction_outputs["instances"].scores[index] > detection_threshold:
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            detected_boxes.append([x1, y1, x2, y2])

    visualize(image, detected_boxes)
