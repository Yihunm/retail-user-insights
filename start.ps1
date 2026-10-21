$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw '请先按 README 创建虚拟环境并安装依赖。'
}
& '.\.venv\Scripts\python.exe' -m streamlit run app.py --server.address 127.0.0.1
