#!/usr/bin/env bash
# ==================== Auto-label 一键安装脚本 ====================
# 适用于 Linux/macOS
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
cd frontend
npm install
cd ..

echo "=========================================="
echo "  安装完成！"
echo ""
echo "  注意：Grounding DINO 模型权重将在首次启动时"
echo "  自动从 HuggingFace Hub 下载，无需手动操作。"
echo ""
echo "  启动命令（4 个终端）："
echo "  1. redis-server"
echo "  2. celery -A backend.worker worker --concurrency=1 --pool=solo -l info"
echo "  3. uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo "  4. cd frontend && npm run dev"
echo "=========================================="
