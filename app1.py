from flask import Flask,jsonify,render_template
import face_recognition
import cv2
import numpy as np



app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/video",methods=['GET'])
def face_video():
    video='Tom Cruise.mp4'
    capture = cv2.VideoCapture('Tom Cruise.mp4') 
    # reading frames
        #ret, frame = webcam.read()
    # Checking if frame captured or not
        #print (ret)

    # displaying image

    if capture.isOpened():
        ret,frame=capture.read()
        cv2.imwrite("faces/tom cruise.jpg",frame)
        load_image=cv2.imread("tom cruise.jpg")
        #cv2.imshow("my image", load_image)
    #releasing the webcam
        capture.release()    
    # stopping the output
        #cv2.waitKey()
    # releasing all windows
        cv2.destroyAllWindows()
    video_capture = cv2.VideoCapture('Tom Cruise.mp4')

    # Load a sample picture and learn how to recognize it.
    gates_image = face_recognition.load_image_file("faces/bill gates.jpg")
    gates_face_encoding = face_recognition.face_encodings(gates_image)[0]

    # Load a second sample picture and learn how to recognize it.
    musk_image = face_recognition.load_image_file("faces/elon musk.jpg")
    musk_face_encoding = face_recognition.face_encodings(musk_image)[0]

    bezos_image = face_recognition.load_image_file("faces/jeff bezos.jpg")
    bezos_face_encoding = face_recognition.face_encodings(bezos_image)[0]

    hanks_image = face_recognition.load_image_file("faces/tom hanks.jpg")
    hanks_face_encoding = face_recognition.face_encodings(hanks_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        gates_face_encoding,
        musk_face_encoding,
        bezos_face_encoding,
        hanks_face_encoding
    ]
    known_face_names = [
        "Bill Gates",
        "Elon Musk",
        "Jeff Bezos",
        "Tom Hanks"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

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

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    print("HI")
    return jsonify(face_names)

@app.route("/webcam",methods=['GET'])
def face_webcam():
    capture = cv2.VideoCapture(0) 
    # reading frames
        #ret, frame = webcam.read()
    # Checking if frame captured or not
        #print (ret)

    # displaying image

    if capture.isOpened():
        ret,frame=capture.read()
        cv2.imwrite("faces/admin.jpg",frame)
        load_image=cv2.imread("admin.jpg")
        #cv2.imshow("my image", load_image)
    #releasing the webcam
        capture.release()    
    # stopping the output
        #cv2.waitKey()
    # releasing all windows
        cv2.destroyAllWindows()
    video_capture = cv2.VideoCapture('Tom Cruise.mp4')

    # Load a sample picture and learn how to recognize it.
    gates_image = face_recognition.load_image_file("faces/bill gates.jpg")
    gates_face_encoding = face_recognition.face_encodings(gates_image)[0]

    # Load a second sample picture and learn how to recognize it.
    musk_image = face_recognition.load_image_file("faces/elon musk.jpg")
    musk_face_encoding = face_recognition.face_encodings(musk_image)[0]

    bezos_image = face_recognition.load_image_file("faces/jeff bezos.jpg")
    bezos_face_encoding = face_recognition.face_encodings(bezos_image)[0]

    hanks_image = face_recognition.load_image_file("faces/tom hanks.jpg")
    hanks_face_encoding = face_recognition.face_encodings(hanks_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        gates_face_encoding,
        musk_face_encoding,
        bezos_face_encoding,
        hanks_face_encoding
    ]
    known_face_names = [
        "Bill Gates",
        "Elon Musk",
        "Jeff Bezos",
        "Tom Hanks"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

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

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return jsonify(face_names)
@app.route("/image")
def face_image():
    return jsonify()

if __name__=="__main__":
    app.run(debug=True)