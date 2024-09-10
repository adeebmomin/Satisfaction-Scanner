import cv2
import random
import pandas as pd
import numpy as np
from deepface import DeepFace
from datetime import datetime
import face_recognition as fr
import openpyxl

def analyze(name):
    # Load the pre-trained emotion detection model
    model = DeepFace.build_model("Emotion")

    # Define emotion labels
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # Load the pre-trained deep learning-based face detection model from OpenCV
    face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

    # Start capturing video
    cap = cv2.VideoCapture(0)

    # Create an empty DataFrame to store satisfaction data
    satisfaction_data = pd.DataFrame(columns=['Timestamp', 'Satisfaction'])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Resize the frame for faster processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Get the height and width of the frame
        (h, w) = frame.shape[:2]

        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the face detector
        face_detector.setInput(blob)
        detections = face_detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                face_roi = frame[y:y1, x:x1]

                resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
                grayscale_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
                normalized_face = grayscale_face / 255.0
                reshaped_face = normalized_face.reshape(1, 48, 48, 1)

                preds = model.predict(reshaped_face)[0]
                emotion_idx = preds.argmax()
                emotion = emotion_labels[emotion_idx]

                if emotion in ['happy', 'surprise']:
                    satisfaction = 0.8
                elif emotion == 'neutral':
                    satisfaction = 0.5
                else:
                    satisfaction = 0.2

                category = 'Satisfied' if satisfaction >= 0.6 else 'Neutral' if satisfaction >= 0.4 else 'Unsatisfied'

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                satisScore = {"Satisfied": 1, "Neutral": .5, "Unsatisfied": 0}

                satisfaction_data.loc[len(satisfaction_data.index)] = [timestamp, satisScore[category]]

                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame, f'{emotion} - {category}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Real-time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

  
    try:
        print('file found')
        #check if there is a sheet that is already named name.
        #if so, simply concat satisfaction_data dataframe to the end of what is already in the excel file.
        #else, create a new sheet in the excel file that just holds the satisfaction_data dataframe
        excel_file_path = 'satisfaction_data.xlsx'
        
        workbook = openpyxl.load_workbook(excel_file_path)

        # Check if the sheet with the specified name already exists
        if name in workbook.sheetnames:
            print('name exists')
            # If the sheet exists, append satisfaction_data to it
            
            existing_df = pd.read_excel(excel_file_path, sheet_name=name)
            combined_df = pd.concat([existing_df, satisfaction_data])

            # Overwrite the existing sheet with the combined data
            print(name)

            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                combined_df.to_excel(writer, sheet_name=name, index=False)

            #combined_df.to_excel(excel_file_path, sheet_name=name, index=False)
         
        else:
            print('name doesnt exist')
            # If the sheet doesn't exist, create a new sheet and write satisfaction_data to it
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
                satisfaction_data.to_excel(writer, sheet_name=name, index=False)
    except FileNotFoundError:
        satisfaction_data.to_excel('satisfaction_data.xlsx', sheet_name=name, index=False)
    

# Call the analyze function with the person's name

