# Очень удобный образ на базе Ubuntu22
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 

# Отключаем лишние вопросы при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive 

# устанавливаем необходимый минимум
RUN apt-get update && apt-get -y install curl cmake build-essential 

WORKDIR /usr/src/onnxruntime-test

# Версия 1.17.3 идеально подходит под наш базовый образ https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-11x
RUN curl -O -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz

COPY . .

ENTRYPOINT ["./run_capi_application.sh", "-p", "onnxruntime-linux-x64-gpu-1.17.3.tgz", "-w", "./"]


