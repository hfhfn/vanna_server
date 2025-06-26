# 使用官方的 Python 基础镜像
FROM python:3.11-slim
#FROM python:3.9-slim-bullseye  # docker线程问题最后应该是这里起作用了（可能跟Ubuntu版本有关系，bullseye这个后缀估计有用）

# 设置工作目录
WORKDIR /app/vanna_server

# 复制 requirements.txt 文件
COPY requirements.txt .

# 安装 Python 依赖
# RUN pip config set global.progress_bar off  # 禁用 pip 的进度条，这个也可能有作用
RUN pip install --no-cache-dir -r requirements.txt

# 安装 pandoc
RUN apt-get update && \
    apt-get install -y pandoc && \
    apt-get clean

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 5000

# 设置 Flask 环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# 启动 Flask 应用
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
