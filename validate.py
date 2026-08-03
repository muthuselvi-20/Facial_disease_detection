import cv2
import numpy as np


def check_brightness(image, threshold=60):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    if brightness < threshold:
        return False, "Move to a brighter place."

    return True, ""


# def check_blur(image, threshold=100):

#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     blur = cv2.Laplacian(gray, cv2.CV_64F).var()

#     if blur < threshold:
#         return False, "Image is blurry."

#     return True, ""


def detect_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return False, "No face detected."

    if len(faces) > 1:
        return False, "Multiple faces detected."

    return True, ""


def validate_image(image):

    for check in [check_brightness, detect_face]:

        valid, message = check(image)

        if not valid:
            return False, message

    return True, "Validation Successful."