import requests
import sys
import os

def test_estimate_height(image_path, reference_height=None, server_url="http://47.96.111.33:8000"):
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
        response = requests.post(url, files=files, data=data)
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python remote_test_height.py <图片路径> [参照物高度cm] [服务地址]")
        print("示例: python remote_test_height.py test_data/fullbody.png 29.7 http://47.96.111.33:8000")
        sys.exit(1)
    image_path = sys.argv[1]
    reference_height = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace('.', '', 1).isdigit() else None
    server_url = sys.argv[3] if len(sys.argv) > 3 else "http://47.96.111.33:8000"
    test_estimate_height(image_path, reference_height, server_url) 