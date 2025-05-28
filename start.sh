#!/bin/bash

# 配置
APP_NAME="face_service"
PORT=8000
DASHSCOPE_API_KEY="sk-87ab9f92e45d40b7ad312f5356fa978c"  # 请替换为实际的API KEY
VENV_PATH="./venv"

# 颜色输出函数
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo_success() {
    echo -e "${GREEN}$1${NC}"
}

echo_error() {
    echo -e "${RED}$1${NC}"
}

# 停止旧进程
stop_old_process() {
    echo "正在检查并停止旧进程..."
    
    # 查找运行在指定端口的进程
    PORT_PID=$(lsof -ti:$PORT)
    if [ ! -z "$PORT_PID" ]; then
        echo "发现端口 $PORT 被占用，进程ID: $PORT_PID"
        kill -15 $PORT_PID
        sleep 2
        
        # 如果进程还在运行，强制终止
        if ps -p $PORT_PID > /dev/null; then
            echo "进程未响应，强制终止..."
            kill -9 $PORT_PID
        fi
        echo_success "旧进程已终止"
    else
        echo "没有发现运行中的旧进程"
    fi
    
    # 查找包含服务名的进程
    pgrep -f "$APP_NAME" | while read -r pid; do
        if [ "$pid" != "$$" ]; then  # 不终止当前脚本进程
            kill -15 $pid
            sleep 1
            if ps -p $pid > /dev/null; then
                kill -9 $pid
            fi
        fi
    done
}

# 启动服务
start_service() {
    echo "正在启动服务..."
    
    # 设置环境变量
    export DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY"
    
    # 激活虚拟环境
    source "$VENV_PATH/bin/activate"
    
    # 使用nohup后台运行服务
    nohup uvicorn $APP_NAME:app --host 0.0.0.0 --port $PORT --reload > face_service.log 2>&1 &
    
    # 获取新进程ID
    NEW_PID=$!
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否成功启动
    if ps -p $NEW_PID > /dev/null; then
        echo_success "服务已成功启动"
        echo_success "进程ID: $NEW_PID"
        echo_success "API地址: http://localhost:$PORT"
        echo_success "日志文件: face_service.log"
    else
        echo_error "服务启动失败，请检查日志文件"
        exit 1
    fi
}

# 主流程
main() {
    echo "=== 人脸服务启动脚本 ==="
    
    # 停止旧进程
    stop_old_process
    
    # 启动服务
    start_service
}

# 运行主流程
main 
