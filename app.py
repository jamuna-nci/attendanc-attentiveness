import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
from ultralytics import YOLO
import time

# Load the models
attendance_model_path = 'mobilenetv2_scripted.pth' 
attendance_model = torch.jit.load(attendance_model_path, map_location=torch.device('cpu'))
attentiveness_model_path = 'best.pt'
attentiveness_model = YOLO(attentiveness_model_path)

# No need to load the state dictionary again since we are using scripted model
attendance_model.eval()

# Load class names
class_names = [
    'Adriana Lima', 'Alex Lawther', 'Alexandra Daddario', 'Alvaro Morte', 'Amanda Crew', 'Andy Samberg',
    'Anne Hathaway', 'Anthony Mackie', 'Avril Lavigne', 'Ben Affleck', 'Bill Gates', 'Bobby Morley',
    'Brenton Thwaites', 'Brian J. Smith', 'Brie Larson', 'Chris Evans', 'Chris Hemsworth', 'Chris Pratt',
    'Christian Bale', 'Cristiano Ronaldo', 'Danielle Panabaker', 'Dominic Purcell', 'Dwayne Johnson',
    'Eliza Taylor', 'Elizabeth Lail', 'Emilia Clarke', 'Emma Stone', 'Emma Watson', 'Gwyneth Paltrow',
    'Henry Cavil', 'Hugh Jackman', 'Inbar Lavi', 'Irina Shayk', 'Jake Mcdorman', 'Jason Momoa',
    'Jennifer Lawrence', 'Jeremy Renner', 'Jessica Barden', 'Jimmy Fallon', 'Johnny Depp', 'Josh Radnor',
    'Katharine Mcphee', 'Katherine Langford', 'Keanu Reeves', 'Krysten Ritter', 'Leonardo DiCaprio',
    'Lili Reinhart', 'Lindsey Morgan', 'Lionel Messi', 'Logan Lerman', 'Madelaine Petsch', 'Maisie Williams',
    'Maria Pedraza', 'Marie Avgeropoulos', 'Mark Ruffalo', 'Mark Zuckerberg', 'Megan Fox', 'Miley Cyrus',
    'Millie Bobby Brown', 'Morena Baccarin', 'Morgan Freeman', 'Nadia Hilker', 'Natalie Dormer',
    'Natalie Portman', 'Neil Patrick Harris', 'Pedro Alonso', 'Penn Badgley', 'Rami Malek', 'Rebecca Ferguson',
    'Richard Harmon', 'Rihanna', 'Robert De Niro', 'Robert Downey Jr', 'Sarah Wayne Callies', 'Selena Gomez',
    'Shakira Isabel Mebarak', 'Sophie Turner', 'Stephen Amell', 'Taylor Swift', 'Tom Cruise', 'Tom Hardy',
    'Tom Hiddleston', 'Tom Holland', 'Tuppence Middleton', 'Ursula Corbero', 'Wentworth Miller', 'Zac Efron',
    'Zendaya', 'Zoe Saldana', 'Alycia Dabnem Carey', 'Amber Heard', 'Barack Obama', 'Barbara Palvin',
    'Camila Mendes', 'Elizabeth Olsen', 'Ellen Page', 'Elon Musk', 'Gal Gadot', 'Grant Gustin', 'Jeff Bezos',
    'Kiernan Shipka', 'Margot Robbie', 'Melissa Fumero', 'Scarlett Johansson', 'Tom Ellis'
]

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess image for attendance
def load_and_preprocess_image_attendance(path):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Load and preprocess image for attentiveness
def load_and_preprocess_image_attentiveness(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Attendance section
def attendance_tracker():
    st.title("Attendance Tracker")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = load_and_preprocess_image_attendance(uploaded_file)
        with torch.no_grad():
            outputs = attendance_model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_names[predicted.item()]
        st.write(f'Predicted: {predicted_label}')
        st.image(uploaded_file, use_container_width=True)

# Attentiveness section
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = attentiveness_model(img_rgb)

    max_conf = -1
    max_label = None
    max_box = None

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confidences):
            if conf > max_conf:
                max_conf = conf
                max_box = box
                max_label = 'Sleepy' if conf > 0.5 else 'Focused'

    if max_box is not None and max_label is not None:
        x1, y1, x2, y2 = map(int, max_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, max_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        if max_label == 'Sleepy':
            st.session_state.sleepy_time += 1

    return av.VideoFrame.from_ndarray(img, format="bgr24")
    
# Main app
def main():
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Attendance Tracker", "Attentiveness Tracker"])
    if app_mode == "Attendance Tracker":
        attendance_tracker()
    elif app_mode == "Attentiveness Tracker":
        attentiveness_tracker()

if __name__ == "__main__":
    main()
