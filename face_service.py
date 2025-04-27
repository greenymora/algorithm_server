from fastapi import FastAPI, UploadFile, File
from typing import Literal
import face_recognition
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io
from fastapi import APIRouter

app = FastAPI()
router = APIRouter(prefix="/face")

# 1. 判断视频中是否包含人脸
def detect_face_in_video(video_path: str) -> bool:
    video = cv2.VideoCapture(video_path)
    has_face = False
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret or frame_count > 30:  # 只检测前30帧，加快速度
            break
        rgb_frame = frame[:, :, ::-1]
        faces = face_recognition.face_locations(rgb_frame)
        if len(faces) > 0:
            has_face = True
            break
        frame_count += 1
    video.release()
    return has_face

# 2. 判断视频与照片是否为同一个人
def is_same_person(video_path: str, image_path: str) -> bool:
    # 1. 提取视频前30帧的人脸特征
    video = cv2.VideoCapture(video_path)
    video_face_encoding = None
    frame_count = 0
    while frame_count < 30:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)
        if encodings:
            video_face_encoding = encodings[0]
            break
    video.release()
    if video_face_encoding is None:
        return False

    # 2. 提取图片人脸特征
    image = face_recognition.load_image_file(image_path)
    image_encodings = face_recognition.face_encodings(image)
    if not image_encodings:
        return False
    image_face_encoding = image_encodings[0]

    # 3. 计算欧氏距离
    distance = np.linalg.norm(video_face_encoding - image_face_encoding)
    return distance < 0.6  # 阈值可调整

# 3. 判断照片类型（人脸照/半身照/全身照）
def classify_photo_type(image_path: str) -> Literal['人脸照', '半身照', '全身照', '无法识别']:
    mp_pose = mp.solutions.pose
    image = cv2.imread(image_path)
    if image is None:
        return '无法识别'
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return '无法识别'
        # 统计可见关键点
        visible = [lm.visibility for lm in results.pose_landmarks.landmark]
        visible_count = sum([v > 0.5 for v in visible])
        if visible_count > 20:
            return '全身照'
        elif visible_count > 10:
            return '半身照'
        else:
            return '人脸照'

@router.post("/has_face_in_video")
def api_has_face_in_video(video: UploadFile = File(...)):
    temp_path = f"/tmp/{video.filename}"
    with open(temp_path, "wb") as f:
        f.write(video.file.read())
    result = detect_face_in_video(temp_path)
    return {"has_face": result}

@router.post("/is_same_person")
def api_is_same_person(video: UploadFile = File(...), image: UploadFile = File(...)):
    temp_video = f"/tmp/{video.filename}"
    temp_image = f"/tmp/{image.filename}"
    with open(temp_video, "wb") as f:
        f.write(video.file.read())
    with open(temp_image, "wb") as f:
        f.write(image.file.read())
    result = is_same_person(temp_video, temp_image)
    return {"is_same_person": bool(result)}

@router.post("/photo_type")
def api_photo_type(image: UploadFile = File(...)):
    temp_image = f"/tmp/{image.filename}"
    with open(temp_image, "wb") as f:
        f.write(image.file.read())
    result = classify_photo_type(temp_image)
    return {"photo_type": result}

app.include_router(router) 