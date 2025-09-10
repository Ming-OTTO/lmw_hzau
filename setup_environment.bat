@echo off
echo 正在配置项目环境...

:: 激活conda环境
call conda activate lmw_python_project

:: 升级pip
python -m pip install --upgrade pip

:: 安装主要依赖
echo 正在安装PaddlePaddle...
pip install paddlepaddle-gpu==2.6.1

echo 正在安装项目依赖...
pip install -r requirements.txt

:: 安装PaddleSeg（从源码）
echo 正在安装PaddleSeg...
cd root\PaddleSeg
pip install -e .

echo.
echo 环境配置完成！
echo 激活环境命令: conda activate lmw_python_project
echo 启动Jupyter: jupyter notebook
echo 启动项目: python 填补v2.py
pause