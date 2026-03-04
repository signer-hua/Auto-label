#!/usr/bin/env bash
# ==================== Auto-label 一键安装脚本 ====================
# 环境要求：Python >= 3.12, CUDA >= 12.6, GPU >= 12GB
set -e

echo "=========================================="
echo "  Auto-label 环境安装脚本"
echo "  技术栈：Grounding DINO + SAM3 + DINOv3"
echo "=========================================="

# 1. 安装 PyTorch (CUDA 12.6)
echo "[1/4] 安装 PyTorch (CUDA 12.6)..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. 安装后端 Python 依赖
echo "[2/4] 安装后端依赖..."
pip install -r backend/requirements.txt

# 3. 安装内置算法库（开发模式）
echo "[3/4] 安装内置算法库 (DINOv3 + SAM3)..."
pip install -e backend/libs/dinov3
pip install -e backend/libs/sam3

# 4. 安装前端依赖
echo "[4/4] 安装前端依赖..."
# 如果存在本地 npm-cache 则使用
NPM_CACHE_DIR="$(dirname "$PWD")/.npm-cache"
if [ -d "$NPM_CACHE_DIR" ]; then
    echo "  使用本地 npm 缓存: $NPM_CACHE_DIR"
    npm config set cache "$NPM_CACHE_DIR"
fi
cd frontend
npm install
cd ..

echo "=========================================="
echo "  安装完成！"
echo ""
echo "  权重文件应放在: ../weights/ 目录下"
echo "    - sam3.pt"
echo "    - dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
echo "    (Grounding DINO 权重首次启动自动下载)"
echo ""
echo "  启动命令（4 个终端）："
echo "  1. redis-server"
echo "  2. celery -A backend.worker worker --concurrency=1 --pool=solo -l info"
echo "  3. uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo "  4. cd frontend && npm run dev"
echo "=========================================="
