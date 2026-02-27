# ==================== 多阶段构建 ====================
# 阶段1：前端构建
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# 阶段2：后端运行环境
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 安装 PyTorch (CUDA 12.1)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 MMCV (YOLO-World 依赖)
RUN pip3 install openmim && mim install mmcv==2.0.0

# 安装后端依赖
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip3 install -r /app/backend/requirements.txt

# 复制算法库
COPY dinov3/ /app/dinov3/
COPY sam3/ /app/sam3/
COPY YOLO-World/ /app/YOLO-World/

# 复制后端代码
COPY backend/ /app/backend/
COPY Auto-label/backend/ /app/Auto-label/backend/

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist /app/Auto-label/frontend/dist

# 创建必要目录
RUN mkdir -p /app/Auto-label/backend/uploads /app/Auto-label/backend/outputs

# 环境变量
ENV PYTHONPATH="/app:/app/Auto-label"
ENV CUDA_VISIBLE_DEVICES=0

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
