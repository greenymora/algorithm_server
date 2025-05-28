import requests
import sys
import os
import time

def test_has_face_in_video(video_path, server_url):
    if not os.path.exists(video_path):
        print(f"文件不存在: {video_path}")
        return
    with open(video_path, 'rb') as f:
        files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
        url = server_url.rstrip('/') + '/face/has_face_in_video'
        print(f"请求地址: {url}")
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

def test_is_same_person(video_path, image_path, server_url):
    if not os.path.exists(video_path) or not os.path.exists(image_path):
        print(f"文件不存在: {video_path} 或 {image_path}")
        return
    with open(video_path, 'rb') as vf, open(image_path, 'rb') as imf:
        files = {
            'video': (os.path.basename(video_path), vf, 'video/mp4'),
            'image': (os.path.basename(image_path), imf, 'image/png' if image_path.endswith('.png') else 'image/jpeg')
        }
        url = server_url.rstrip('/') + '/face/is_same_person'
        print(f"请求地址: {url}")
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

def test_photo_type(image_path, server_url):
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/png' if image_path.endswith('.png') else 'image/jpeg')}
        url = server_url.rstrip('/') + '/face/photo_type'
        print(f"请求地址: {url}")
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

def test_photo_taken_time(image_path, server_url):
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/png' if image_path.endswith('.png') else 'image/jpeg')}
        url = server_url.rstrip('/') + '/face/photo_taken_time'
        print(f"请求地址: {url}")
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

# 新增的身高估算接口测试函数
def test_estimate_height(image_path, reference_height=None, server_url="http://localhost:8000"):
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return
    url = server_url.rstrip('/') + '/face/estimate_height'
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/png' if image_path.endswith('.png') else 'image/jpeg')}
        data = {}
        if reference_height is not None:
            data['reference_height'] = str(reference_height)
        print(f"请求地址: {url}")
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        end_time = time.time()
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

def test_is_same_face(image1_path, image2_path, server_url):
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print(f"文件不存在: {image1_path} 或 {image2_path}")
        return
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        files = {
            'image1': (os.path.basename(image1_path), f1, 'image/jpeg'),
            'image2': (os.path.basename(image2_path), f2, 'image/jpeg')
        }
        url = server_url.rstrip('/') + '/face/is_same_face'
        print(f"请求地址: {url}")
        start_time = time.time()
        response = requests.post(url, files=files)
        end_time = time.time()
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    print(f"运行时间: {end_time - start_time:.2f} 秒")

def test_extract_face_from_oss_video(oss_url, server_url):
    url = server_url.rstrip('/') + '/face/extract_face_from_oss_video'
    data = {'oss_url': oss_url}
    print(f"请求地址: {url}")
    print(f"OSS视频URL: {oss_url}")
    print("（本地无法直接测试OSS视频，如需测试请提供可访问的OSS视频URL）")
    # response = requests.post(url, data=data)
    # print(f"响应状态码: {response.status_code}")
    # print(f"响应内容: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法:")
        print("  python remote_test_face.py has_face_in_video <视频路径> [服务地址]")
        print("  python remote_test_face.py is_same_person <视频路径> <图片路径> [服务地址]")
        print("  python remote_test_face.py photo_type <图片路径> [服务地址]")
        print("  python remote_test_face.py photo_taken_time <图片路径> [服务地址]")
        print("  python remote_test_face.py estimate_height <图片路径> [参照物高度cm] [服务地址]")
        print("  python remote_test_face.py is_same_face <图片1路径> <图片2路径> [服务地址]")
        print("  python remote_test_face.py extract_face_from_oss_video <oss视频url> [服务地址]")
        sys.exit(1)
    interface = sys.argv[1]
    server_url = sys.argv[-1] if sys.argv[-1].startswith('http') else "http://localhost:8000"

    if interface == 'has_face_in_video':
        video_path = sys.argv[2]
        test_has_face_in_video(video_path, server_url)
    elif interface == 'is_same_person':
        if len(sys.argv) < 4:
            print("is_same_person 需要视频路径和图片路径")
            sys.exit(1)
        video_path = sys.argv[2]
        image_path = sys.argv[3]
        test_is_same_person(video_path, image_path, server_url)
    elif interface == 'photo_type':
        image_path = sys.argv[2]
        test_photo_type(image_path, server_url)
    elif interface == 'photo_taken_time':
        image_path = sys.argv[2]
        test_photo_taken_time(image_path, server_url)
    elif interface == 'estimate_height': # 处理新的身高估算接口
        if len(sys.argv) < 3:
            print("estimate_height 需要图片路径")
            sys.exit(1)
        image_path = sys.argv[2]
        # 检查是否有参照物高度和服务地址参数
        reference_height = None
        current_index = 3
        if len(sys.argv) > current_index and sys.argv[current_index].replace('.', '', 1).isdigit():
            reference_height = float(sys.argv[current_index])
            current_index += 1
        
        if len(sys.argv) > current_index and sys.argv[current_index].startswith('http'):
            server_url = sys.argv[current_index]
        else:
             server_url = "http://47.96.111.33:8000"

        test_estimate_height(image_path, reference_height, server_url)
    elif interface == 'is_same_face':
        # 示例：python remote_test_face.py is_same_face test_data/1.jpg test_data/2.jpg [服务地址]
        if len(sys.argv) < 4:
            print("is_same_face 需要两张图片路径")
            sys.exit(1)
        image1_path = sys.argv[2]
        image2_path = sys.argv[3]
        test_is_same_face(image1_path, image2_path, server_url)
    elif interface == 'extract_face_from_oss_video':
        # 示例：python remote_test_face.py extract_face_from_oss_video <oss视频url> [服务地址]
        if len(sys.argv) < 3:
            print("extract_face_from_oss_video 需要OSS视频URL")
            sys.exit(1)
        oss_url = sys.argv[2]
        test_extract_face_from_oss_video(oss_url, server_url)
    else:
        print(f"未知接口: {interface}")
        sys.exit(1) 