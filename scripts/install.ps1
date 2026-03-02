# ==================== Auto-label Windows 安装脚本 ====================
# 环境要求：Python >= 3.12, CUDA >= 12.6, GPU >= 12GB
# 权重文件目录：e:\auto-label\weights\
# Redis 目录：E:\app\Redis\Redis-x64-5.0.14.1
# npm 缓存目录：e:\auto-label\.npm-cache
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Auto-label 环境安装脚本 (Windows)" -ForegroundColor Cyan
Write-Host "  技术栈：Grounding DINO + SAM3 + DINOv3" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. 安装 PyTorch (CUDA 12.6)
Write-Host "[1/4] 安装 PyTorch (CUDA 12.6)..." -ForegroundColor Yellow
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. 安装后端依赖
Write-Host "[2/4] 安装后端依赖..." -ForegroundColor Yellow
pip install -r backend\requirements.txt

# 3. 安装内置算法库
Write-Host "[3/4] 安装内置算法库 (DINOv3 + SAM3)..." -ForegroundColor Yellow
pip install -e backend\libs\dinov3
pip install -e backend\libs\sam3

# 4. 安装前端依赖（使用本地 npm-cache）
Write-Host "[4/4] 安装前端依赖..." -ForegroundColor Yellow
$npmCacheDir = Join-Path (Split-Path $PSScriptRoot -Parent) ".npm-cache"
if (Test-Path $npmCacheDir) {
    Write-Host "  使用本地 npm 缓存: $npmCacheDir" -ForegroundColor Gray
    npm config set cache $npmCacheDir
}
Set-Location frontend
npm install
Set-Location ..

Write-Host "==========================================" -ForegroundColor Green
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "  权重文件应放在: ..\weights\ 目录下" -ForegroundColor Green
Write-Host "    - sam3.pt" -ForegroundColor Green
Write-Host "    - dinov3_vits16_pretrain_lvd1689m-08c60483.pth" -ForegroundColor Green
Write-Host "    (Grounding DINO 权重首次启动自动下载)" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "  启动命令（4 个终端）：" -ForegroundColor Green
Write-Host "  1. E:\app\Redis\Redis-x64-5.0.14.1\redis-server.exe" -ForegroundColor Green
Write-Host "  2. celery -A backend.worker worker --concurrency=1 --pool=solo -l info" -ForegroundColor Green
Write-Host "  3. uvicorn backend.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Green
Write-Host "  4. cd frontend; npm run dev" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
