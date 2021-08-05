#Created by:Arjith
#Date:22-06-2021
#Purpose: Real Time face Recognition and training
from flask import Flask,jsonify,render_template
import face_recognition
import cv2
import numpy as np
import os
known_face_names=[]
known_face_encodings=[]

for filename in os.listdir('faces/'):
    new_image=face_recognition.load_image_file('faces/'+filename)
    print(filename)
    new_image_encoding=face_recognition.face_encodings(new_image)[0]
    known_face_names.append(new_image)
    known_face_encodings.append(new_image_encoding)

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/video",methods=['GET'])
def face_video():
    video='Tom Cruise.mp4'
    
    video_capture = cv2.VideoCapture('Tom Cruise.mp4')

    # Load a sample picture and learn how to recognize it.
    

    # Create arrays of known face encodings and their names
    

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    name="Unknown"
    fr=[]
    n=0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        fr=frame
        

        

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    if name=='Unknown':
        val=input("Enter the name")
        val=str(val)
        cv2.imwrite('faces/'+val+'.jpg',fr)
        new_image = face_recognition.load_image_file("faces/"+val+".jpg")
        new_face_encoding=face_recognition.face_encodings(new_image)[0]
        known_face_names.append(val)
        known_face_encodings.append(new_face_encoding)
    return jsonify(face_names)

@app.route("/webcam",methods=['GET'])
def face_webcam():
    
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    

    # Create arrays of known face encodings and their names
    

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    name="Unknown"
    fr=[]
    n=0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        fr=frame
        

        

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    if name=='Unknown':
        val=input("Enter the name")
        val=str(val)
        cv2.imwrite('faces/'+val+'.jpg',fr)
        new_image = face_recognition.load_image_file("faces/"+val+".jpg")
        new_face_encoding=face_recognition.face_encodings(new_image)[0]
        known_face_names.append(val)
        known_face_encodings.append(new_face_encoding)
    return jsonify(face_names)
@app.route("/image")
def face_image():
    return jsonify()

if __name__=="__main__":
    app.run(debug=True)
