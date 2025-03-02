# 第一阶段：构建依赖
FROM python:3.9-slim as builder
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --user numpy==1.23.5 torch==2.0.1 torchaudio==2.0.2 speechbrain

# 第二阶段：精简镜像
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY voice_filter_container.py .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "voice_filter_container.py"]
