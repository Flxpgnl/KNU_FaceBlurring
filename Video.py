import cv2
import dlib
import numpy as np
import time
import psutil
import os

def detect_and_blur_faces_live():
    face_net = cv2.dnn.readNetFromCaffe(
        "Model/deploy.prototxt",
        "Model/res10_300x300_ssd_iter_140000.caffemodel"
    )
    predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(1)

    frame_count = 0
    start_time = time.time()
    process = psutil.Process(os.getpid())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = dlib.rectangle(startX, startY, endX, endY)
                landmarks = predictor(gray, face)

                points = []
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    points.append((x, y))

                forehead_height = int((endY - startY) * 0.45)
                for i in range(0, 17):
                    x = landmarks.part(i).x
                    y = max(0, landmarks.part(i).y - forehead_height)
                    points.append((x, y))
                    
                side_expansion = int((endX - startX) * 0.2)
                left_points = points[:9]  # Defining left part of face (landmarks 0-8)
                right_points = points[8:17]  # Defining right part of face (landmarks 8-16)

                for x, y in left_points:
                    points.append((max(0, x - side_expansion), y))
                for x, y in right_points:
                    points.append((min(w, x + side_expansion), y))

                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)

        blurred = cv2.GaussianBlur(frame, (151, 151), 50)
        output = np.where(mask[:,:,None] == 255, blurred, frame)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate and display memory usage
        memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        cv2.putText(output, f"Memory: {memory_usage:.2f} MB", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Blurred Faces', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate average FPS
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

detect_and_blur_faces_live()
detect_and_blur_faces_live()
