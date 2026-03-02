# ==================== Auto-label Windows 一键启动脚本 ====================
# 在一个 PowerShell 窗口中启动所有 4 个服务
# 使用方法：在 Auto-label 目录下执行  .\scripts\start.ps1
param(
    [string]$RedisPath = "E:\app\Redis\Redis-x64-5.0.14.1\redis-server.exe"
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Auto-label 启动脚本" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 检查 Redis
if (-not (Test-Path $RedisPath)) {
    Write-Host "[错误] Redis 未找到: $RedisPath" -ForegroundColor Red
    Write-Host "  请修改 -RedisPath 参数指向正确的 redis-server.exe" -ForegroundColor Red
    exit 1
}

# 设置环境变量
$env:PYTHONPATH = (Get-Location).Path

Write-Host "[1/4] 启动 Redis..." -ForegroundColor Yellow
Start-Process -FilePath $RedisPath -WindowStyle Minimized

Start-Sleep -Seconds 2

Write-Host "[2/4] 启动 Celery Worker..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; `$env:PYTHONPATH='$PWD'; celery -A backend.worker worker --concurrency=1 --pool=solo -l info"

Start-Sleep -Seconds 2

Write-Host "[3/4] 启动 FastAPI..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; `$env:PYTHONPATH='$PWD'; uvicorn backend.main:app --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 2

Write-Host "[4/4] 启动前端..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm run dev"

Write-Host "==========================================" -ForegroundColor Green
Write-Host "  所有服务已启动！" -ForegroundColor Green
Write-Host "  前端地址: http://localhost:5173" -ForegroundColor Green
Write-Host "  后端地址: http://localhost:8000" -ForegroundColor Green
Write-Host "  API 文档: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
