# Squeezenet in Docker using ORT-CUDA (C++)


## Установка и запуск
Предполагается, что у вас уже имеется установленный Docker. Также предполагается, что вы используете Linux, все ссылки ведут на документацию для Linux.
1. Установить [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html). Обязательно выполнить все шаги по подготовке и пост-установке. 

2. Установить [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Обязательно выполнить все шаги по подготовке и пост-установке, [прописать config.json](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) для докера с рантаймом нвидии. 

3.  Подготовить докер образ
    - Сбилдить образ самому: 
        ``` 
        docker build testovoe-cuda:latest .
        ```
    - Или воспользоваться готовым.
      
        - Тяжелая (8GB) CUDA версия:
            ``` 
            docker pull swedenmoose/testovoe-cuda:latest
            ```
        - Lightweight (485MB) CPU версия:
            ``` 
            docker pull swedenmoose/testovoe-cpu:latest
            ```

4. Запустить образ
    ```
    docker run --rm -it --runtime=nvidia --gpus all swedenmoose/testovoe-cuda:latest
    ```
    или
    ```
    docker run --rm -it swedenmoose/testovoe-cpu:latest
    ```
5. Фиксировать результат, выполненный с использованием всей мощи GPU.
![Вся мощь GPU](/docs/images/cuda-result.jpeg)
