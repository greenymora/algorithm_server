from fastapi import FastAPI, UploadFile, File
from typing import Literal, Optional
import face_recognition
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io
from fastapi import APIRouter
from PIL.ExifTags import TAGS
from datetime import datetime
import paddleocr
import dashscope
from dashscope import MultiModalConversation
import re
import os
from fastapi.responses import JSONResponse

app = FastAPI()
router = APIRouter(prefix="/face")

# 初始化PaddleOCR
paddle_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")

def standardize_time_format(time_str: str) -> Optional[str]:
    """将各种格式的时间字符串转换为标准格式 (YYYY-MM-DD HH:mm:ss)"""
    if not time_str:
        return None
        
    # 定义可能的时间格式
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y年%m月%d日 %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y年%m月%d日 %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y.%m.%d %H:%M",
        "%Y-%m-%d",
        "%Y年%m月%d日",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%H:%M:%S",
        "%H:%M"
    ]
    
    # 尝试解析时间字符串
    parsed_time = None
    for fmt in formats:
        try:
            parsed_time = datetime.strptime(time_str.strip(), fmt)
            break
        except ValueError:
            continue
            
    # 如果只有时间没有日期，使用当前日期
    if parsed_time and len(time_str) <= 8 and ":" in time_str:  # 处理纯时间格式
        current_date = datetime.now()
        parsed_time = parsed_time.replace(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
    
    # 如果没有时间，补充为当天0点
    if parsed_time and parsed_time.hour == 0 and parsed_time.minute == 0 and parsed_time.second == 0:
        if len(time_str) <= 10:  # 只有日期的情况
            current_time = datetime.now()
            parsed_time = parsed_time.replace(
                hour=current_time.hour,
                minute=current_time.minute,
                second=current_time.second
            )
    
    # 返回标准格式的时间字符串
    return parsed_time.strftime("%Y-%m-%d %H:%M:%S") if parsed_time else None

def extract_time_from_text(text: str) -> Optional[str]:
    """从文本中提取时间信息"""
    # 匹配常见的时间格式
    patterns = [
        # 完整的日期时间格式
        r'(\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{1,2}(?::\d{1,2})?)',  # 2024年1月1日 12:00:00
        r'(\d{4}-\d{1,2}-\d{1,2}\s*\d{1,2}:\d{1,2}(?::\d{1,2})?)',      # 2024-1-1 12:00:00
        r'(\d{4}/\d{1,2}/\d{1,2}\s*\d{1,2}:\d{1,2}(?::\d{1,2})?)',      # 2024/1/1 12:00:00
        r'(\d{4}\.\d{1,2}\.\d{1,2}\s*\d{1,2}:\d{1,2}(?::\d{1,2})?)',    # 2024.1.1 12:00:00
        
        # 日期和时间分开的格式
        r'(\d{4}年\d{1,2}月\d{1,2}日).*?(\d{1,2}:\d{1,2}(?::\d{1,2})?)', # 匹配分开的日期和时间
        
        # 仅日期格式
        r'(\d{4}年\d{1,2}月\d{1,2}日)',
        r'(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{4}/\d{1,2}/\d{1,2})',
        r'(\d{4}\.\d{1,2}\.\d{1,2})',
        
        # 仅时间格式
        r'(\d{1,2}:\d{1,2}(?::\d{1,2})?)'
    ]
    
    found_dates = []
    found_times = []
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if '年' in pattern or '/' in pattern or '-' in pattern or '.' in pattern:
                # 如果是完整的日期时间
                if ':' in match.group(1):
                    return standardize_time_format(match.group(1))
                # 如果是日期
                found_dates.append(match.group(1))
            elif ':' in pattern:
                # 如果是时间
                found_times.append(match.group(1))
            # 处理日期和时间分开的情况
            if match.lastindex == 2:
                date = match.group(1)
                time = match.group(2)
                # 将日期和时间组合
                return standardize_time_format(f"{date} {time}")
    
    # 如果找到了日期和时间，尝试组合它们
    if found_dates and found_times:
        # 优先使用最长的时间格式（可能包含秒）
        longest_time = max(found_times, key=len)
        return standardize_time_format(f"{found_dates[0]} {longest_time}")
    
    # 如果只找到日期，返回标准化的日期
    if found_dates:
        return standardize_time_format(found_dates[0])
        
    # 如果只找到时间，使用当前日期
    if found_times:
        longest_time = max(found_times, key=len)
        return standardize_time_format(longest_time)
    
    return None

async def get_time_from_qwen(image_path: str) -> Optional[str]:
    """使用通义千问多模态模型识别图片中的时间信息"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        messages = [{
            'role': 'user',
            'content': [{
                'text': '请帮我识别这张图片中的拍摄时间或者显示的时间信息，只需要返回时间，不需要其他描述。如果有多个时间，请返回最可能的拍摄时间。如果没有找到时间信息，请返回"未找到时间信息"',
                'image': image_data
            }]
        }]
        
        response = await MultiModalConversation.acreate(
            model='qwen-vl-plus',
            messages=messages,
            api_key=os.getenv('DASHSCOPE_API_KEY', '')
        )
        
        if response.output and response.output.choices:
            time_str = response.output.choices[0].message.content
            if time_str != "未找到时间信息":
                return standardize_time_format(time_str)
        return None
    except Exception as e:
        print(f"通义千问API调用错误: {str(e)}")
        return None

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

# 4. 获取图片拍摄时间
async def get_photo_taken_time(image_path: str) -> dict:
    """获取照片拍摄时间，支持多种识别方式"""
    try:
        # 1. 首先尝试从EXIF中获取时间
        image = Image.open(image_path)
        exif = image.getexif()
        if exif:
            for tag_id in exif:
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTime" or tag == "DateTimeOriginal" or tag == "DateTimeDigitized":
                    date_str = exif.get(tag_id)
                    try:
                        taken_time = standardize_time_format(date_str)
                        if taken_time:
                            return {
                                "has_exif": True,
                                "taken_time": taken_time,
                                "error": None,
                                "method": "exif"
                            }
                    except Exception as e:
                        return {
                            "has_exif": True,
                            "taken_time": None,
                            "error": f"日期格式解析错误: {str(e)}",
                            "method": "exif"
                        }

        # 2. 如果没有EXIF，尝试使用OCR识别
        ocr_result = paddle_ocr.ocr(image_path, cls=True)
        if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
            # 将所有识别到的文本合并
            all_texts = []
            for line in ocr_result[0]:
                text = line[1][0]
                all_texts.append(text)
            
            text = " ".join(all_texts)
            time_str = extract_time_from_text(text)
            if time_str:
                return {
                    "has_exif": False,
                    "taken_time": time_str,
                    "error": None,
                    "method": "ocr",
                    "raw_text": text
                }
        
        # 3. 如果OCR也没识别到，使用通义千问
        qwen_result = await get_time_from_qwen(image_path)
        if qwen_result:
            return {
                "has_exif": False,
                "taken_time": qwen_result,
                "error": None,
                "method": "qwen"
            }

        return {
            "has_exif": False,
            "taken_time": None,
            "error": "无法识别到任何时间信息",
            "method": None
        }
        
    except Exception as e:
        return {
            "has_exif": False,
            "taken_time": None,
            "error": str(e),
            "method": None
        }

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

@router.post("/photo_taken_time")
async def api_photo_taken_time(image: UploadFile = File(...)):
    try:
        temp_image = f"/tmp/{image.filename}"
        contents = await image.read()
        with open(temp_image, "wb") as f:
            f.write(contents)
            
        result = await get_photo_taken_time(temp_image)
        
        if os.path.exists(temp_image):
            os.remove(temp_image)
            
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

app.include_router(router) 