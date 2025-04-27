# face_service 算法服务

## 功能简介
本服务基于 FastAPI，提供以下三个人脸/人体相关算法接口：

1. 判断视频中是否包含人脸
2. 判断视频中的人脸与照片是否为同一个人
3. 根据照片判断是人脸照、半身照还是全身照

---

## 依赖安装
建议使用 Python 3.7 及以上版本。

```bash
pip install fastapi uvicorn face_recognition opencv-python mediapipe pillow
```

> 注意：`mediapipe` 仅支持 Python 3.7 及以上。

---

## 启动方法

```bash
uvicorn face_service:app --reload
```

---

## 接口说明

### 1. 判断视频中是否包含人脸
- 路径：`/has_face_in_video`
- 方法：POST
- 参数：上传视频文件（form-data，字段名为`video`）
- 返回：
  ```json
  { "has_face": true }
  ```

### 2. 判断视频与照片是否为同一个人
- 路径：`/is_same_person`
- 方法：POST
- 参数：上传视频文件（字段名`video`）和照片文件（字段名`image`）
- 返回：
  ```json
  { "is_same_person": true }
  ```

### 3. 根据照片判断人脸照、半身照、全身照
- 路径：`/photo_type`
- 方法：POST
- 参数：上传照片文件（字段名`image`）
- 返回：
  ```json
  { "photo_type": "人脸照" }
  ```

---

## 其他说明
- 上传文件会临时保存到`/tmp`目录，处理后自动释放。
- 如需扩展更多算法接口，可在`face_service.py`中添加。 