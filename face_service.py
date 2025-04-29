def standardize_time_format(time_str: str) -> str:
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
    
    if not parsed_time:
        return None
        
    # 如果只有时间没有日期，使用当前日期
    if len(time_str) <= 8 and ":" in time_str:  # 处理纯时间格式
        current_date = datetime.now()
        parsed_time = parsed_time.replace(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
    
    # 如果没有时间，补充为当天0点
    if parsed_time.hour == 0 and parsed_time.minute == 0 and parsed_time.second == 0:
        if len(time_str) <= 10:  # 只有日期的情况
            current_time = datetime.now()
            parsed_time = parsed_time.replace(
                hour=current_time.hour,
                minute=current_time.minute,
                second=current_time.second
            )
    
    # 返回标准格式的时间字符串
    return parsed_time.strftime("%Y-%m-%d %H:%M:%S")

def extract_time_from_text(text: str) -> str:
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

async def get_photo_taken_time(image_path: str) -> dict:
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
                        # 标准化EXIF时间格式
                        taken_time = standardize_time_format(date_str)
                        if taken_time:
                            return {
                                "has_exif": True,
                                "taken_time": taken_time,
                                "error": None,
                                "method": "exif"
                            }
                    except Exception as e:
                        return {"has_exif": True, "taken_time": None, "error": f"日期格式解析错误: {str(e)}", "method": "exif"}

        # 2. 如果没有EXIF，尝试使用OCR识别
        ocr_result = paddle_ocr.ocr(image_path, cls=True)
        if ocr_result:
            # 将所有识别到的文本合并
            all_texts = []
            for line in ocr_result[0]:
                text = line[1][0]
                all_texts.append(text)
                print(f"OCR识别文本: {text}")  # 调试输出
            
            text = " ".join(all_texts)
            time_str = extract_time_from_text(text)
            if time_str:
                return {
                    "has_exif": False,
                    "taken_time": time_str,
                    "error": None,
                    "method": "ocr",
                    "raw_text": text  # 添加原始识别文本用于调试
                }
        
        # 3. 如果OCR也没识别到，使用通义千问
        qwen_result = await get_time_from_qwen(image_path)
        if qwen_result and qwen_result != "未找到时间信息":
            # 标准化通义千问返回的时间格式
            standardized_time = standardize_time_format(qwen_result)
            if standardized_time:
                return {
                    "has_exif": False,
                    "taken_time": standardized_time,
                    "error": None,
                    "method": "qwen"
                }

        return {
            "has_exif": False,
            "taken_time": None,
            "error": "无法识别到任何时间信息",
            "method": None,
            "raw_text": text if 'text' in locals() else None
        }
    except Exception as e:
        return {
            "has_exif": False,
            "taken_time": None,
            "error": str(e),
            "method": None
        } 