#!/usr/bin/env bash
# ==================== Auto-label 一键安装脚本 ====================
# 适用于 Linux/macOS，Windows 请参考 README.md 手动安装
set -e

echo "=========================================="
echo "  Auto-label 环境安装脚本"
echo "=========================================="

# 1. 安装 PyTorch (CUDA 12.1)
echo "[1/5] 安装 PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安装 MMCV (YOLO-World 依赖，必须通过 openmim 安装)
echo "[2/5] 安装 MMCV..."
pip install openmim
mim install mmcv==2.0.0

# 3. 安装后端 Python 依赖
echo "[3/5] 安装后端依赖..."
pip install -r backend/requirements.txt

# 4. 安装内置算法库（开发模式）
echo "[4/5] 安装内置算法库..."
pip install -e backend/libs/dinov3
pip install -e backend/libs/sam3
pip install -e backend/libs/yolo_world

# 5. 安装前端依赖
echo "[5/5] 安装前端依赖..."
cd frontend
npm install
cd ..

echo "=========================================="
echo "  安装完成！"
echo "  启动后端: python -m backend.main"
echo "  启动前端: cd frontend && npm run dev"
echo "=========================================="
