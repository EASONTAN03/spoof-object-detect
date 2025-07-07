import streamlit as st
import os
from ultralytics import YOLO
import cv2

bbox_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
               (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 0),
               (128, 128, 0), (0, 0, 128)]

def display_prediction(predicted_image):
    
    if predicted_image is not None:
        st.image(predicted_image, caption="Predicted Image", use_container_width=True)
    else:
        st.warning("No objects detected.")

def spoof_detect(model, labels, img_path, min_conf, min_iou):
    try:
        # Read image
        frame = cv2.imread(img_path)

        # Run prediction
        results = model(img_path, conf=min_conf, iou=min_iou, agnostic_nms=True, verbose=False)
        detections = results[0].boxes

        if len(detections) > 0:
            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)

                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                conf = detections[i].conf.item()
                color = bbox_colors[classidx % len(bbox_colors)]

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Convert to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            return None  # no detections

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def process_image_input():
    model_type = st.radio("Choose models:", ("v1.1"))
    
    if model_type == "v1.1":
        model = YOLO('../spoof-detect_v1.1_1750604360.pt', task='detect')  # Adjust path as needed
    labels = model.names

    min_conf = st.slider(
        "Confidence Threshold (conf):", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.45, 
        step=0.01
    )

    # IoU slider
    min_iou = st.slider(
        "IoU Threshold (iou):", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.8, 
        step=0.01
    )

    uploaded_file = st.file_uploader(f"Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner('Predicting...'):
            # Save the uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", width=200)

            with col2:
                predicted = spoof_detect(model, labels, "temp_image.jpg", min_conf, min_iou)
                display_prediction(predicted)

            # Remove the temporary file
            os.remove("temp_image.jpg")