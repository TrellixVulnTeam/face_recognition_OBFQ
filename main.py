import numpy as np
import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)


param_image = face_recognition.load_image_file("/Users/param/Downloads/param.png")
param_face_encoding = face_recognition.face_encodings(param_image)[0]





known_face_encodings = [param_face_encoding]

known_face_names = ["param"]



while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        results = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_result_index = np.argmin(face_distances)
        if results[best_result_index]:
            name = known_face_names[best_result_index]


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), (2))

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 225), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)



    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


