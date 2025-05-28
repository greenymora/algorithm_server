from fastapi import FastAPI, UploadFile, File, Form
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
from py_iztro import Astro
import traceback
import subprocess
import json
import sys
import base64
from io import BytesIO
from PIL import Image as PILImage
import requests

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
    # 1. 提取视频前15帧的人脸特征 (优化)
    video = cv2.VideoCapture(video_path)
    video_face_encoding = None
    frame_count = 0
    max_frames_to_check = 15 # 修改为检查前15帧
    target_width = 320 # 缩放目标宽度

    while frame_count < max_frames_to_check:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

        # 缩放帧 (优化)
        img_h, img_w = frame.shape[:2]
        if img_w > target_width:
            scale = target_width / img_w
            frame = cv2.resize(frame, (target_width, int(img_h * scale)))

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

@router.post("/estimate_height")
async def estimate_height(
    image: UploadFile = File(...),
    reference_height: Optional[float] = Form(None)  # 参照物实际高度（cm），可选
):
    temp_image = f"/tmp/{image.filename}"
    try:
        with open(temp_image, "wb") as f:
            f.write(await image.read())
        # 读取图片
        img = cv2.imread(temp_image)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "图片读取失败"})
        # Mediapipe人体关键点检测
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                return {"error": "未检测到人体关键点，无法估算身高"}
            # 获取头顶和脚底关键点
            landmarks = results.pose_landmarks.landmark
            # 头顶：NOSE(0)或LEFT_EYE(1)/RIGHT_EYE(2)附近最高点
            # 脚底：LEFT_HEEL(27), RIGHT_HEEL(28)
            y_top = min([landmarks[0].y, landmarks[1].y, landmarks[2].y])
            y_bottom = max([landmarks[27].y, landmarks[28].y])
            img_h, img_w = img.shape[:2]
            pixel_height = (y_bottom - y_top) * img_h
            # 估算身高
            if reference_height and reference_height > 0:
                return {
                    "pixel_height": pixel_height,
                    "reference_height": reference_height,
                    "estimate_height": None,
                    "confidence": 0.0,
                    "algorithm": "需要参照物像素高度，建议后续扩展自动检测或让用户标注",
                    "error": "暂未实现参照物像素高度自动检测，请输入参照物像素高度后再试"
                }
            else:
                estimate_height = pixel_height / img_h * 170
                return {
                    "pixel_height": pixel_height,
                    "estimate_height": round(estimate_height, 2),
                    "confidence": 0.5,
                    "algorithm": "无参照物，采用标准人体比例估算，误差较大，仅供参考",
                    "error": None
                }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_image):
            os.remove(temp_image)

def format_star_list(star_list):
    return '、'.join([
        f"{star.name}{'('+star.brightness+')' if star.brightness else ''}{'['+star.mutagen+']' if star.mutagen else ''}"
        for star in star_list
    ]) if star_list else '无'

def ziwei_full_chart_str(
    gender: str,
    date_type: str,  # "公历" 或 "农历"
    date_str: str,   # "YYYY-MM-DD"
    hour_index: int  # 0~11，子时=0，丑时=1...，午时=6
):
    print(f"[DEBUG] 调用ziwei_full_chart_str: gender={gender}, date_type={date_type}, date_str={date_str}, hour_index={hour_index}")
    try:
        astro = Astro()
        print("[DEBUG] Astro对象创建成功")
        if date_type == "公历":
            print("[DEBUG] 调用astro.by_solar")
            result = astro.by_solar(date_str, hour_index, gender)
        elif date_type == "农历":
            print("[DEBUG] 调用astro.by_lunar")
            result = astro.by_lunar(date_str, hour_index, gender)
        else:
            raise ValueError("date_type 只能为 '公历' 或 '农历'")
        print(f"[DEBUG] 排盘主结果: {result}")
        today = datetime.today().strftime("%Y-%m-%d")
        print(f"[DEBUG] today={today}")
        horoscope = result.horoscope(today)
        print(f"[DEBUG] horoscope={horoscope}")
        lines = []
        lines.append(f"===== 紫微斗数排盘结果（{date_type} {date_str}，{['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥'][hour_index]}时，{gender}）=====")
        lines.append(f"命盘公历生日: {result.solar_date}")
        lines.append(f"命盘农历生日: {result.lunar_date}")
        lines.append(f"四柱: {result.chinese_date}")
        lines.append(f"生肖: {result.zodiac}  星座: {result.sign}")
        lines.append(f"命宫: {result.earthly_branch_of_soul_palace}  身宫: {result.earthly_branch_of_body_palace}")
        lines.append(f"命主: {result.soul}  身主: {result.body}")
        lines.append(f"五行局: {result.five_elements_class}")
        lines.append("")
        for i in range(12):
            palace = result.palaces[i]
            lines.append(
                f"宫位: {palace.name}\n"
                f"  干支: {palace.heavenly_stem}{palace.earthly_branch}\n" 
                f"  主星: {format_star_list(palace.major_stars)}\n"
                f"  辅星: {format_star_list(palace.minor_stars)}\n"
                f"  杂曜: {format_star_list(palace.adjective_stars)}\n"
                f"  大运: 大运{horoscope.decadal.palace_names[i]}\n"
                f"    大运星: {format_star_list(horoscope.decadal.stars[i])}\n"
                f"  流年: 流年{horoscope.yearly.palace_names[i]}\n"
                f"    流年星: {format_star_list(horoscope.yearly.stars[i])}\n"
                f"  流月: 流月{horoscope.monthly.palace_names[i]}\n"
                f"    流月星: {format_star_list(horoscope.monthly.stars[i])}\n"
                f"  流日: 流日{horoscope.daily.palace_names[i]}\n"
                f"    流日星: {format_star_list(horoscope.daily.stars[i])}\n"
                f"  流时: 流时{horoscope.hourly.palace_names[i]}\n"
                f"    流时星: {format_star_list(horoscope.hourly.stars[i])}\n"
            )
        print("[DEBUG] 排盘文本生成完毕")
        return '\n'.join(lines)
    except Exception as e:
        print("[ERROR] 紫微排盘接口异常：", repr(e))
        print("[ERROR] Traceback:\n" + traceback.format_exc())
        return JSONResponse(status_code=400, content={"error": str(e), "traceback": traceback.format_exc()})

@router.post("/ziwei_chart")
def api_ziwei_chart(
    birthday: str = Form(..., description="生日，格式YYYY-MM-DD"),
    birth_time: str = Form(..., description="出生时间，格式HH:MM"),
    gender: str = Form(..., description="性别，男或女"),
    date_type: str = Form("公历", description="历法类型，公历或农历")
):
    try:
        hour, minute = map(int, birth_time.split(":"))
        hour_index = (hour + 1) // 2 % 12
        # 调用子进程
        cmd = [
            sys.executable,  # 当前python解释器
            "/root/workspace/algorithm_server/ziwei_cli.py",
            gender,
            date_type,
            birthday,
            str(hour_index)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": "子进程执行失败", "detail": result.stderr})
        try:
            data = json.loads(result.stdout)
        except Exception:
            return JSONResponse(status_code=500, content={"error": "子进程输出无法解析", "raw": result.stdout})
        return data
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

def get_largest_face_encoding(image_path):
    img = face_recognition.load_image_file(image_path)
    # 先缩放图片（如宽度>600则缩放）
    h, w = img.shape[:2]
    if w > 600:
        scale = 600 / w
        img = cv2.resize(img, (600, int(h * scale)))
    # 检测所有人脸
    locations = face_recognition.face_locations(img)
    if not locations:
        return None
    # 选最大人脸
    areas = [(b-t)*(r-l) for (t, r, b, l) in locations]
    max_idx = areas.index(max(areas))
    encoding = face_recognition.face_encodings(img, [locations[max_idx]])
    return encoding[0] if encoding else None

@router.post("/is_same_face")
def api_is_same_face(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    try:
        temp1 = f"/tmp/{image1.filename}"
        temp2 = f"/tmp/{image2.filename}"
        with open(temp1, "wb") as f:
            f.write(image1.file.read())
        with open(temp2, "wb") as f:
            f.write(image2.file.read())
        enc1 = get_largest_face_encoding(temp1)
        enc2 = get_largest_face_encoding(temp2)
        if enc1 is None or enc2 is None:
            return {"is_same_face": False, "error": "未检测到人脸"}
        distance = float(np.linalg.norm(enc1 - enc2))
        is_same = distance < 0.6
        return {"is_same_face": bool(is_same), "distance": distance}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp1):
            os.remove(temp1)
        if os.path.exists(temp2):
            os.remove(temp2)

@router.post("/extract_face_from_oss_video")
def extract_face_from_oss_video(
    oss_url: str = Form(..., description="阿里OSS视频URL")
):
    """
    从阿里OSS视频中抽取人物照片并以base64返回
    """
    try:
        # 下载视频到本地临时文件
        temp_video = "/tmp/oss_video.mp4"
        with requests.get(oss_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_video, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # 读取视频帧，检测人脸
        video = cv2.VideoCapture(temp_video)
        found_face = False
        face_img_b64 = None
        for _ in range(60):  # 检查前60帧
            ret, frame = video.read()
            if not ret:
                break
            rgb_frame = frame[:, :, ::-1]
            faces = face_recognition.face_locations(rgb_frame)
            if faces:
                top, right, bottom, left = faces[0]
                face_img = frame[top:bottom, left:right]
                pil_img = PILImage.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                face_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                found_face = True
                break
        video.release()
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if found_face:
            return {"face_base64": face_img_b64}
        else:
            return JSONResponse(status_code=404, content={"error": "未检测到人脸"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

app.include_router(router) 