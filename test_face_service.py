import requests
import os
from datetime import datetime

BASE_URL = 'http://127.0.0.1:8000/face'

def test_has_face_in_video(video_path):
    """测试视频人脸检测接口"""
    url = f'{BASE_URL}/has_face_in_video'
    with open(video_path, 'rb') as f:
        files = {'video': f}
        resp = requests.post(url, files=files)
    print(f'测试 {video_path} 是否有人脸: ', resp.json())

def test_is_same_person(video_path, image_path):
    """测试人脸比对接口"""
    url = f'{BASE_URL}/is_same_person'
    with open(video_path, 'rb') as vf, open(image_path, 'rb') as imgf:
        files = {'video': vf, 'image': imgf}
        resp = requests.post(url, files=files)
    print(f'测试 {video_path} 和 {image_path} 是否同一人: ', resp.json())

def test_photo_type(image_path):
    """测试照片类型识别接口"""
    url = f'{BASE_URL}/photo_type'
    with open(image_path, 'rb') as f:
        files = {'image': f}
        resp = requests.post(url, files=files)
    print(f'测试 {image_path} 类型: ', resp.json())

def test_photo_taken_time(image_path):
    """测试照片拍摄时间接口"""
    url = f'{BASE_URL}/photo_taken_time'
    with open(image_path, 'rb') as f:
        files = {'image': f}
        resp = requests.post(url, files=files)
    result = resp.json()
    
    # 打印详细信息
    print(f'\n测试 {image_path} 拍摄时间:')
    print(f'  识别方法: {result.get("method", "未知")}')
    print(f'  是否有EXIF: {result.get("has_exif", False)}')
    print(f'  拍摄时间: {result.get("taken_time", "未知")}')
    if result.get("error"):
        print(f'  错误信息: {result.get("error")}')
    if result.get("raw_text"):
        print(f'  OCR原始文本: {result.get("raw_text")}')
    print()

def verify_time_freshness(image_path, max_days=7):
    """验证照片的时间新鲜度"""
    url = f'{BASE_URL}/photo_taken_time'
    with open(image_path, 'rb') as f:
        files = {'image': f}
        resp = requests.post(url, files=files)
    result = resp.json()
    
    if result.get("taken_time"):
        try:
            # 解析时间字符串（标准格式：YYYY-MM-DD HH:mm:ss）
            taken_time = datetime.strptime(result["taken_time"], "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            time_diff = now - taken_time
            
            print(f'\n验证 {image_path} 时间新鲜度:')
            print(f'  拍摄时间: {result["taken_time"]}')
            print(f'  当前时间: {now.strftime("%Y-%m-%d %H:%M:%S")}')
            print(f'  时间差: {time_diff.days}天 {time_diff.seconds//3600}小时')
            
            if time_diff.days > max_days:
                print(f'  结论: 照片可能过期（超过{max_days}天）')
            else:
                print(f'  结论: 照片时间在有效期内')
        except Exception as e:
            print(f'\n验证 {image_path} 时间新鲜度: 解析错误 - {str(e)}')
    else:
        print(f'\n验证 {image_path} 时间新鲜度: 未找到时间信息')
    print()

def prepare_test_data():
    """准备测试数据目录"""
    test_dir = 'test_data'
    os.makedirs(test_dir, exist_ok=True)
    
    # 检查测试文件是否存在
    required_files = ['1.mp4', '2.mp4', '1.jpg', '2.jpg']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(test_dir, f))]
    
    if missing_files:
        print("\n警告：以下测试文件不存在：")
        for file in missing_files:
            print(f"  - test_data/{file}")
        print("\n请确保将测试文件放在 test_data 目录下")
        return False
    return True

if __name__ == '__main__':
    print("=== 开始测试 ===")
    
    # 准备测试数据
    if not prepare_test_data():
        print("\n测试终止：缺少测试文件")
        exit(1)
    
    try:
        # 测试视频人脸检测
        print("\n1. 测试视频人脸检测")
        test_has_face_in_video('test_data/1.mp4')
        test_has_face_in_video('test_data/2.mp4')
        
        # 测试视频和图片是否同一人
        print("\n2. 测试人脸比对")
        test_is_same_person('test_data/1.mp4', 'test_data/1.jpg')
        test_is_same_person('test_data/2.mp4', 'test_data/2.jpg')
        
        # 测试图片类型
        print("\n3. 测试照片类型")
        test_photo_type('test_data/1.jpg')
        test_photo_type('test_data/2.jpg')
        
        # 测试照片拍摄时间
        print("\n4. 测试照片拍摄时间")
        test_photo_taken_time('test_data/1.jpg')
        test_photo_taken_time('test_data/2.jpg')
        
        # 验证照片时间新鲜度
        print("\n5. 验证照片新鲜度")
        verify_time_freshness('test_data/1.jpg')
        verify_time_freshness('test_data/2.jpg')
        
        print("\n=== 测试完成 ===")
        
    except requests.exceptions.ConnectionError:
        print("\n错误：无法连接到服务器，请确保服务已启动")
        print(f"当前服务地址：{BASE_URL}")
        print("提示：检查 start.sh 是否正常运行")
    except Exception as e:
        print(f"\n测试过程中出现错误：{str(e)}") 