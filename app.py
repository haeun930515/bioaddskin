from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from skimage import feature
from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
import os

app = Flask(__name__)

# Mediapipe FaceMesh 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# 모델 로드
MODEL_PATH = 'more_data(3).h5'
new_model = load_model(MODEL_PATH)

def compute_lbp_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 24
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    return np.sum(lbp_hist)

def draw_landmarks_with_flicker(image):
    results = face_mesh.process(image)
    landmarks_image = np.zeros_like(image, dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]

                start_point = (int(landmarks.landmark[start_idx].x * image.shape[1]),
                               int(landmarks.landmark[start_idx].y * image.shape[0]))
                end_point = (int(landmarks.landmark[end_idx].x * image.shape[1]),
                             int(landmarks.landmark[end_idx].y * image.shape[0]))

                cv2.line(landmarks_image, start_point, end_point, (220, 220, 220), 1, lineType=cv2.LINE_AA)
                
                # Draw the landmark points
                cv2.circle(landmarks_image, start_point, 1, (127, 127, 127), -1)

    # Now, apply a slight blur to make the lines appear thinner
    landmarks_image = cv2.GaussianBlur(landmarks_image, (3, 3), 0)
    
    # Blend the original image with the landmarks image for a translucent effect
    alpha = 0.35
    blended_image = cv2.addWeighted(image, 1 - alpha, landmarks_image, alpha, 0)
    
    return blended_image

def count_wrinkles_and_spots(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray_roi, 9, 80, 80)
    edges = cv2.Canny(bilateral, 50, 150)
    
    wrinkles = np.sum(edges > 0)
    
    # Use adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to fill small holes and remove small noises
    kernel = np.ones((3,3), np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours to reduce noise
    min_spot_area = 4
    spots = len([cnt for cnt in contours if cv2.contourArea(cnt) > min_spot_area])
    
    return wrinkles, spots

def count_features(image):
    wrinkles, spots = count_wrinkles_and_spots(image)
    texture = compute_lbp_texture(image)
    return wrinkles, spots, texture

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    frame = np.array(image)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    wrinkles, spots, texture = count_features(frame)
    frame = draw_landmarks_with_flicker(frame)

    return frame, wrinkles, spots, texture

def loadImage(filepath):
    test_img = tf_image.load_img(filepath, target_size=(180, 180))
    test_img = tf_image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255
    return test_img


def model_predict(img_path):
    global new_model
    age_pred = new_model.predict(loadImage(img_path))
    x = age_pred[0][0]
    rounded_age_value = round(x)  # Rounds 24.56 to 25
    age = str(rounded_age_value)
    return age

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # POST 요청에서 이미지 파일 가져오기
        image_file = request.files['image']
        
        # 업로드된 파일을 임시로 저장
        temp_path = "temp.jpg"
        image_file.save(temp_path)

        # 이미지 처리
        frame, wrinkles, spots, texture = process_image(temp_path)

        # 나이 예측 수행
        age = model_predict(temp_path)

        # 분석 결과 반환
        result = {
            "wrinkles": int(wrinkles),
            "spots": int(spots),
            "texture": float(texture),  # 예시로 float으로 변환
            "age": age
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
