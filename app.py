import cv2
import streamlit as st
from deepface import DeepFace


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a Streamlit app
st.title("Facial Expression Detector")
flag = False
button_col1, button_col2 = st.columns(2)

# url = 'https://192.168.1.3:8080/video'

# Open a webcam
cap = cv2.VideoCapture(0)

if button_col1.button("Start Webcam",key='start_button'):
    flag = True
if button_col2.button("Stop Webcam",key='stop_button'):
    flag = False
    cap.release()
    # cv2.destroyAllWindows()

emotions_window = st.empty()
FRAME_WINDOW = st.image([])


webcam_text_window = st.empty()
if flag == False:
    webcam_text_window.text("Please Start Webcam")

if flag == True:
    emotions_window.text("No Face Detected")

# Check if the webcam is opened successfully
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
elif not cap.isOpened():   
    st.error("Failed to open webcam")
    st.stop()

# Loop to continuously capture frames from the webcam
while flag:
    while True and flag:
        ret, frame = cap.read()
        try:
            result = DeepFace.analyze(frame, actions = ['emotion'])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,1.1,4)

            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame,
                #             result[0]
                #     ['dominant_emotion'],
                #     (0,50),
                #         font,1,
                #         (0,0,255),
                #         2,
                #         cv2.LINE_4
                #     )
                  
            emotions_window.text("Facial Expressions Detected: " +result[0]['dominant_emotion'].upper())
            
        
        except:
            # text_window.text("Emotions Detected")
            emotions_window.text("No Face Detected")
            if flag==False:
                break
            pass

        if flag==False:
            break
        FRAME_WINDOW.image(frame,channels="BGR", use_column_width=True)

