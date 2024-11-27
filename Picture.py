import cv2
import dlib
import numpy as np

def detect_and_blur_faces(image_path):
    # Load face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Create a mask for blurring
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Create a convex hull around the face
        points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
        
        hull = cv2.convexHull(np.array(points))
        
        # Draw filled polygon on the mask
        cv2.fillConvexPoly(mask, hull, 255)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (99, 99), 30)

    # Blend the blurred image with the original image using the mask
    output = np.where(mask[:,:,None] == 255, blurred, image)

    # Display the result
    cv2.imshow('Blurred Faces', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite('blurred_faces_contour.jpg', output)

    # Print the number of faces detected
    print(f"Number of faces detected and blurred: {len(faces)}")

# Specify the path to your image
image_path = 'FaceRefCover1.jpg.webp'

# Call the function to detect and blur faces
detect_and_blur_faces(image_path)
