import os
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
from flask import Flask, render_template, request, redirect, url_for, flash, Response
import time
from datetime import datetime
import telepot

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize the Telegram bot with your bot's API token
bot = telepot.Bot('6679358098:AAFmpDc7o4MwqDywDahyAK0Qq89IVZqNr04')  # Replace 'YOUR_BOT_API_TOKEN' with your actual token

# Global variables for video processing
model = None
Q = deque(maxlen=128)
(W, H) = (None, None)
violence_detected = False
violence_start_frame = None
frame_count = 0

# Function to process frames and send alerts
def process_frame(frame):
    global model, Q, W, H, violence_detected, violence_start_frame, frame_count
    
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255

    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    results = np.array(Q).mean(axis=0)
    i = (preds > 0.50)[0]
    label = i

    text_color = (0, 255, 0) 

    if label:
        text_color = (0, 0, 255)
        
        if not violence_detected:
            violence_detected = True
            violence_start_frame = frame_count
            violence_start_time = time.time()
    else:
        violence_detected = False

    if violence_detected and frame_count == violence_start_frame + 30:
        # Capture the current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Send the alert with timestamp to the Telegram group
        message = f"Violence detected at {current_time}"
        with open('alert_frame.jpg', 'wb') as f:
            cv2.imwrite('alert_frame.jpg', frame * 255)
            bot.sendPhoto(telegram_group_id, open('alert_frame.jpg', 'rb'), caption=message)

    text = "Violence: {}".format(label)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

    return output

# Route for video upload
@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    global model, Q, W, H, violence_detected, violence_start_frame, frame_count
    
    if request.method == 'POST':
        file = request.files['file']

        if file:
            filename = 'V68.mp4'
            file.save(filename)

            # Initialize the video capture
            vs = cv2.VideoCapture(filename)

            # Define the codec and create a VideoWriter object for output
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (1280, 720))  # Adjust the resolution as needed

            while True:
                (grabbed, frame) = vs.read()

                if not grabbed:
                    break

                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                output = process_frame(frame)

                # Write the frame with annotations to the output video
                out.write(output)

                cv2.imshow("Violence Detection", output)

                key = cv2.waitKey(1) & 0xFF
                frame_count += 1

                if key == ord("q"):
                    break

            vs.release()
            out.release()
            cv2.destroyAllWindows()

    return render_template('upload.html')

# Route for real-time webcam detection
@app.route('/webcam')
def webcam_detection():
    global model, Q, W, H, violence_detected, violence_start_frame, frame_count
    
    # Initialize the video capture for webcam
    vs = cv2.VideoCapture(0)  # 0 represents the default camera

    # Define the codec and create a VideoWriter object for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('webcam_output.avi', fourcc, 30.0, (1280, 720))  # Adjust the resolution as needed

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = process_frame(frame)

        # Write the frame with annotations to the output video
        out.write(output)

        cv2.imshow("Violence Detection", output)

        key = cv2.waitKey(1) & 0xFF
        frame_count += 1

        if key == ord("q"):
            break

    vs.release()
    out.release()
    cv2.destroyAllWindows()
    return render_template('index.html')


# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Load the model and initialize any other necessary variables
    model = load_model('modelnew.h5')
    
    # Replace 'YOUR_TELEGRAM_GROUP_ID' with your Telegram group ID
    telegram_group_id = '-949413618'
    
    app.run(debug=True)
