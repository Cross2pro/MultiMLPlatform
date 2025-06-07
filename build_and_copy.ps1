# 设置控制台编码为UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "============================================" -ForegroundColor Green
Write-Host "自动编译和部署脚本" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

try {
    Write-Host "[1/4] 进入前端项目目录..." -ForegroundColor Yellow
    
    if (!(Test-Path "ml-prediction-system")) {
        throw "错误：ml-prediction-system 目录不存在"
    }
    
    Set-Location "ml-prediction-system"
    
    Write-Host "[2/4] 开始编译前端项目..." -ForegroundColor Yellow
    Write-Host "正在运行 npm run build..." -ForegroundColor Cyan
    
    $buildResult = & npm run build
    if ($LASTEXITCODE -ne 0) {
        throw "错误：前端项目编译失败"
    }
    
    Write-Host "[3/4] 清空目标目录..." -ForegroundColor Yellow
    Set-Location ".."
    
    $targetDir = "ml-prediction-system-backend-flask\app\static\frontend"
    
    if (Test-Path $targetDir) {
        Write-Host "正在清空 $targetDir 目录..." -ForegroundColor Cyan
        Remove-Item $targetDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "[4/4] 复制编译文件到目标目录..." -ForegroundColor Yellow
    Write-Host "正在复制文件从 ml-prediction-system\out 到 $targetDir..." -ForegroundColor Cyan
    
    $sourceDir = "ml-prediction-system\out"
    if (!(Test-Path $sourceDir)) {
        throw "错误：源目录 $sourceDir 不存在，请确认编译是否成功"
    }
    
    # 创建目标目录（如果不存在）
    if (!(Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }
    
    # 复制所有文件和子目录
    Copy-Item -Path "$sourceDir\*" -Destination $targetDir -Recurse -Force
    
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "编译和部署完成！" -ForegroundColor Green
    Write-Host "源目录：$sourceDir" -ForegroundColor White
    Write-Host "目标目录：$targetDir" -ForegroundColor White
    Write-Host "============================================" -ForegroundColor Green
    
} catch {
    Write-Host "错误：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    # 回到原始目录
    Set-Location $PSScriptRoot
    Write-Host "按任意键继续..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 