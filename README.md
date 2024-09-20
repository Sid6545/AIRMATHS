# AIRMATHS
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector 
import google.generativeai as genai 
from PIL import Image
import streamlit as st  

st.set_page_config(layout="wide")
st.image('mathgesture.png')
col1, col2 = st.columns([3,2])
with col1:
    run = st.checkbox('Run',value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("answer")
    output_yext_area = st.subheader("")


genai.configure(api_key="AIzaSyB8Bu9BjCbMWUKhVTt_9OrspkUgBmsTJ4Q")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    # Check if the index finger is up (fingers = [0, 1, 0, 0, 0])
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(lmList[8][0:2])  # Convert to tuple for cv2.line
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, current_pos, color=(255, 0, 255), thickness=10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)

    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math Problem", pil_image])
        return response.text

prev_pos = None
canvas = None
image_combined = None
output_text = ""     #None can also written like ""

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally for a mirror effect

    if not success:
        break

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info  # Extract fingers and landmark list from info
        prev_pos, canvas = draw(info, prev_pos, canvas)  # Correctly unpack the return values
        output_text = sendToAI(model, canvas, fingers)

    # Blend the image and canvas using addWeighted
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined,channels="BGR")

    if output_text:
        output_yext_area.text(output_text)

    # Display the image, canvas, and the combined image
    # cv2.imshow("Image", img)
    #cv2.imshow("Canvas", canvas)
    #cv2.imshow("Image Combined", image_combined)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
