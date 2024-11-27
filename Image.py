import cv2
import dlib
import numpy as np

def detect_and_blur_faces_image(image_path):
    face_net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not read the image.")
        return

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
    cv2.imshow('Blurred Faces in Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to your image
detect_and_blur_faces_image('FaceRefCover1.jpg.webp')