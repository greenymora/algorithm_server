from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import os
from typing import Optional

app = FastAPI()

@app.post("/face/estimate_height")
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
                # 需要用户输入参照物像素高度，或自动检测（可扩展）
                # 这里假设参照物为A4纸，用户输入实际高度
                # 让用户在图片中用A4纸靠近身体，后续可扩展自动检测A4纸
                # 这里暂不实现自动检测，直接用用户输入
                # 让用户用画图工具标注参照物像素高度也可
                # 这里假设参照物像素高度与人体像素高度相同
                # 实际应用中建议让用户输入参照物像素高度
                # 这里直接返回提示
                return {
                    "pixel_height": pixel_height,
                    "reference_height": reference_height,
                    "estimate_height": None,
                    "confidence": 0.0,
                    "algorithm": "需要参照物像素高度，建议后续扩展自动检测或让用户标注",
                    "error": "暂未实现参照物像素高度自动检测，请输入参照物像素高度后再试"
                }
            else:
                # 无参照物，采用人体标准比例估算（误差较大）
                # 参考文献：成年男性平均头身比约1:7.5，女性约1:7.3
                # 这里假设头顶到脚底像素高度约等于身高
                # 经验值：像素高度/人体实际身高=常数K（需大量样本标定，这里假设K=1）
                # 这里假设图片为标准全身照，人体直立无遮挡
                # 以170cm为基准，像素高度为图片高度的90%时，身高约170cm
                # 估算公式：estimate_height = pixel_height / img_h * 170
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