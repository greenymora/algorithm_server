# 人脸识别服务

这是一个基于FastAPI的人脸识别服务，提供人脸检测、人脸比对、照片类型识别和拍摄时间验证等功能。

## 环境要求

- Python 3.8+
- 依赖包：
  ```
  fastapi
  uvicorn
  face-recognition
  paddleocr
  dashscope
  python-multipart
  ```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置环境变量：
```bash
export DASHSCOPE_API_KEY='your_api_key_here'
```

3. 启动服务：
```bash
./start.sh
```

## API 接口说明

所有接口的基础URL为：`http://your-server:8000/face`

### 1. 视频人脸检测

检测视频中是否包含人脸。

- **接口**：`/has_face_in_video`
- **方法**：POST
- **参数**：
  - video: 视频文件（multipart/form-data）
- **返回示例**：
```json
{
    "has_face": true
}
```

### 2. 人脸比对

比对视频和照片中的人脸是否为同一人。

- **接口**：`/is_same_person`
- **方法**：POST
- **参数**：
  - video: 视频文件（multipart/form-data）
  - image: 图片文件（multipart/form-data）
- **返回示例**：
```json
{
    "is_same_person": true
}
```

### 3. 照片类型识别

识别照片类型（人脸照/半身照/全身照）。

- **接口**：`/photo_type`
- **方法**：POST
- **参数**：
  - image: 图片文件（multipart/form-data）
- **返回示例**：
```json
{
    "photo_type": "人脸照"  // 可能的值：人脸照、半身照、全身照、无法识别
}
```

### 4. 照片拍摄时间验证

获取照片的拍摄时间信息，支持多种识别方式。所有时间格式都会被标准化为 "YYYY-MM-DD HH:mm:ss" 格式。

- **接口**：`/photo_taken_time`
- **方法**：POST
- **参数**：
  - image: 图片文件（multipart/form-data）
- **返回示例**：
```json
{
    "has_exif": true,
    "taken_time": "2024-03-15 14:30:00",
    "error": null,
    "method": "exif",  // 可能的值：exif、ocr、qwen
    "raw_text": "原始识别文本（仅OCR时提供）"
}
```

识别方法优先级：
1. EXIF信息（最可靠）
2. OCR文字识别
3. 通义千问多模态模型

支持的输入时间格式：
```
完整格式：
- 2024-03-15 14:30:00
- 2024年3月15日 14:30:00
- 2024/3/15 14:30:00
- 2024.3.15 14:30:00

日期和时间分开：
- 2024-03-15 14:30
- 2024年3月15日 14:30
- 2024/3/15 14:30
- 2024.3.15 14:30

仅日期（将使用当前时间）：
- 2024-03-15
- 2024年3月15日
- 2024/3/15
- 2024.3.15

仅时间（将使用当前日期）：
- 14:30:00
- 14:30
```

时间处理规则：
1. 如果只有日期没有时间，将使用当前时间
2. 如果只有时间没有日期，将使用当前日期
3. 所有时间格式都会被标准化为 "YYYY-MM-DD HH:mm:ss"
4. 如果时间格式无法解析，将返回错误信息

## 测试

使用测试脚本进行功能测试：

```bash
python test_face_service.py
```

测试脚本会对所有接口进行测试，并验证照片的时间新鲜度。

## 错误处理

所有接口在发生错误时会返回相应的错误信息：
- 400：请求参数错误
- 500：服务器内部错误

错误响应格式：
```json
{
    "error": "错误描述信息"
}
```

## 注意事项

1. 确保服务器有足够的内存运行模型
2. 视频检测默认只检查前30帧以提高性能
3. 人脸比对使用欧氏距离阈值0.6
4. 照片时间验证建议结合业务需求设置合适的时效性阈值
5. 使用通义千问API需要确保有足够的调用额度
6. 所有返回的时间格式均为标准的 "YYYY-MM-DD HH:mm:ss" 格式

## 开发计划

- [ ] 添加人脸特征提取API
- [ ] 支持更多的时间格式识别
- [ ] 优化视频处理性能
- [ ] 添加批量处理接口
- [ ] 增加缓存机制

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

MIT License 