<#
Auto-label Windows Startup Script (适配Conda虚拟环境)
Conda环境名称：auto-label
使用方法：在Auto-label根目录执行 .\scripts\start.ps1
#>
param(
    [string]$RedisPath = "E:\app\Redis\Redis-x64-5.0.14.1\redis-server.exe",
    [string]$CondaEnvName = "auto-label",  # Conda虚拟环境名称
    [string]$CondaPath = "H:\Anaconda\Scripts\conda.exe"  # Conda可执行文件路径
)

# 打印标题
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Auto-label Startup Script (Conda Env)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 检查Redis路径是否存在
if (-not (Test-Path -Path $RedisPath -PathType Leaf)) {
    Write-Host "[ERROR] Redis not found: $RedisPath" -ForegroundColor Red
    Write-Host "  Fix: Use -RedisPath to specify correct redis-server.exe path" -ForegroundColor Red
    exit 1
}

# 检查Conda是否存在
if (-not (Test-Path -Path $CondaPath -PathType Leaf)) {
    Write-Host "[ERROR] Conda not found: $CondaPath" -ForegroundColor Red
    Write-Host "  Fix: Modify -CondaPath to your actual conda.exe path" -ForegroundColor Red
    exit 1
}

# 获取项目根路径（绝对路径）
$rootPath = (Get-Location).Path
$env:PYTHONPATH = $rootPath

# 1. 启动Redis
Write-Host "[1/4] Starting Redis..." -ForegroundColor Yellow
try {
    Start-Process -FilePath $RedisPath -WindowStyle Minimized -ErrorAction Stop
    Write-Host "[SUCCESS] Redis started successfully" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Redis start failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
Start-Sleep -Seconds 2

# 2. 启动Celery Worker (激活Conda环境后执行)
Write-Host "[2/4] Starting Celery Worker (Conda env: $CondaEnvName)..." -ForegroundColor Yellow
$celeryCmd = @"
cd '$rootPath';
# 初始化Conda并激活虚拟环境
& '$CondaPath' init powershell;
& conda activate $CondaEnvName;
# 设置环境变量并启动Celery
`$env:PYTHONPATH='$rootPath';
python -m celery -A backend.worker worker --concurrency=1 --pool=solo -l info;
"@
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy Bypass", "-Command", $celeryCmd
Start-Sleep -Seconds 2

# 3. 启动FastAPI (激活Conda环境后执行)
Write-Host "[3/4] Starting FastAPI (Conda env: $CondaEnvName)..." -ForegroundColor Yellow
$fastapiCmd = @"
cd '$rootPath';
# 初始化Conda并激活虚拟环境
& '$CondaPath' init powershell;
& conda activate $CondaEnvName;
# 设置环境变量并启动FastAPI
`$env:PYTHONPATH='$rootPath';
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000;
"@
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy Bypass", "-Command", $fastapiCmd
Start-Sleep -Seconds 2

# 4. 启动前端
Write-Host "[4/4] Starting Frontend..." -ForegroundColor Yellow
$frontendPath = Join-Path -Path $rootPath -ChildPath "frontend"
if (-not (Test-Path -Path $frontendPath)) {
    Write-Host "[WARNING] Frontend folder not found: $frontendPath" -ForegroundColor Yellow
}
$frontendCmd = @"
cd '$frontendPath';
npm run dev;
"@
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy Bypass", "-Command", $frontendCmd

# 启动完成提示
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  All services startup commands sent!" -ForegroundColor Green
Write-Host "  Conda Env: $CondaEnvName" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "  Backend: http://localhost:8000" -ForegroundColor Green
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green