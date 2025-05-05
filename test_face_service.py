import pytest
import os
import requests
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import numpy as np
from datetime import datetime
import piexif

# 测试服务器地址
BASE_URL = "http://localhost:8000/face"

# 测试资源目录
TEST_RESOURCES_DIR = "test_resources"

def setup_module(module):
    """创建测试资源目录"""
    if not os.path.exists(TEST_RESOURCES_DIR):
        os.makedirs(TEST_RESOURCES_DIR)
        
    # 创建测试图片
    create_test_image_with_time()
    create_test_video_with_face()
    create_test_image_with_face()

def create_test_image_with_time():
    """创建带有时间水印的测试图片"""
    # 创建一个渐变背景
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    for i in range(600):
        img[i, :] = [i//3, i//3, i//3]
    
    # 添加当前时间水印
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    
    # 在图片上添加多个位置的时间信息
    cv2.putText(img, current_time, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"拍摄时间：{current_time}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"摄影时间：{current_time}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存图片
    cv2.imwrite(f"{TEST_RESOURCES_DIR}/test_image_with_time.jpg", img)
    
    # 添加EXIF信息
    # 创建EXIF数据
    exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
    
    # 添加DateTime标签
    current_time = datetime.now().strftime("%Y:%m:%d %H:%M:%S")  # EXIF日期格式
    exif_dict["0th"][piexif.ImageIFD.DateTime] = current_time
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = current_time
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = current_time
    
    # 将EXIF数据写入图片
    exif_bytes = piexif.dump(exif_dict)
    im = Image.open(f"{TEST_RESOURCES_DIR}/test_image_with_time.jpg")
    im.save(f"{TEST_RESOURCES_DIR}/test_image_with_time.jpg", exif=exif_bytes)

def create_test_video_with_face():
    """创建包含人脸的测试视频"""
    out = cv2.VideoWriter(
        f"{TEST_RESOURCES_DIR}/test_video.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (640, 480)
    )
    
    # 创建10帧的视频
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 在帧中间画一个简单的圆形作为"人脸"
        cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)
        out.write(frame)
    
    out.release()

def create_test_image_with_face():
    """创建包含人脸的测试图片"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 在图片中间画一个简单的圆形作为"人脸"
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
    cv2.imwrite(f"{TEST_RESOURCES_DIR}/test_face.jpg", img)

def test_has_face_in_video():
    """测试视频人脸检测接口"""
    with open(f"{TEST_RESOURCES_DIR}/test_video.mp4", 'rb') as f:
        files = {'video': ('test_video.mp4', f, 'video/mp4')}
        response = requests.post(f"{BASE_URL}/has_face_in_video", files=files)
        
    assert response.status_code == 200
    result = response.json()
    assert 'has_face' in result
    assert isinstance(result['has_face'], bool)

def test_is_same_person():
    """测试人脸比对接口"""
    with open(f"{TEST_RESOURCES_DIR}/test_video.mp4", 'rb') as video_file, \
         open(f"{TEST_RESOURCES_DIR}/test_face.jpg", 'rb') as image_file:
        files = {
            'video': ('test_video.mp4', video_file, 'video/mp4'),
            'image': ('test_face.jpg', image_file, 'image/jpeg')
        }
        response = requests.post(f"{BASE_URL}/is_same_person", files=files)
        
    assert response.status_code == 200
    result = response.json()
    assert 'is_same_person' in result
    assert isinstance(result['is_same_person'], bool)

def test_photo_type():
    """测试照片类型识别接口"""
    with open(f"{TEST_RESOURCES_DIR}/test_face.jpg", 'rb') as f:
        files = {'image': ('test_face.jpg', f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/photo_type", files=files)
        
    assert response.status_code == 200
    result = response.json()
    assert 'photo_type' in result
    assert result['photo_type'] in ['人脸照', '半身照', '全身照', '无法识别']

def test_photo_taken_time():
    """测试照片拍摄时间识别接口"""
    test_image = "test_data/image.png"
    
    with open(test_image, 'rb') as f:
        files = {'image': ('image.png', f, 'image/png')}
        response = requests.post(f"{BASE_URL}/photo_taken_time", files=files)
    
    print(f"\n响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    assert response.status_code == 200
    result = response.json()
    assert 'taken_time' in result
    assert 'method' in result
    assert 'error' in result
    
    if result['taken_time']:
        # 验证返回的时间格式是否正确
        try:
            datetime.strptime(result['taken_time'], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pytest.fail("Invalid datetime format")

def teardown_module(module):
    """清理测试资源"""
    import shutil
    if os.path.exists(TEST_RESOURCES_DIR):
        shutil.rmtree(TEST_RESOURCES_DIR)

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 