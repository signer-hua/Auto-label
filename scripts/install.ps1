# ==================== Auto-label Windows 安装脚本 ====================
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Auto-label 环境安装脚本 (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. 安装 PyTorch (CUDA 12.1)
Write-Host "[1/5] 安装 PyTorch..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安装 MMCV
Write-Host "[2/5] 安装 MMCV..." -ForegroundColor Yellow
pip install openmim
mim install mmcv==2.0.0

# 3. 安装后端依赖
Write-Host "[3/5] 安装后端依赖..." -ForegroundColor Yellow
pip install -r backend\requirements.txt

# 4. 安装内置算法库
Write-Host "[4/5] 安装内置算法库..." -ForegroundColor Yellow
pip install -e backend\libs\dinov3
pip install -e backend\libs\sam3
pip install -e backend\libs\yolo_world

# 5. 安装前端依赖
Write-Host "[5/5] 安装前端依赖..." -ForegroundColor Yellow
Set-Location frontend
npm install
Set-Location ..

Write-Host "==========================================" -ForegroundColor Green
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "  启动后端: python -m backend.main" -ForegroundColor Green
Write-Host "  启动前端: cd frontend; npm run dev" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
