@echo off
echo 正在激活项目环境...
call conda activate lmw_python_project
echo 环境已激活！
echo.
echo 当前Python路径：
python --version
echo.
echo 常用命令：
echo   运行填补v2.py: python 填补v2.py
echo   启动Jupyter: jupyter notebook
echo   查看环境包: conda list
echo.
cmd /k