#!/bin/bash
ENV_NAME="diplom_env"

if ! command -v conda &> /dev/null; then
    echo "Conda не установлена"
    exit 1
fi

echo "Создание виртуального окружения $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

pip install pandas numpy torch scikit-learn darts statsmodels matplotlib

if [ $? -eq 0 ]; then
    echo "Успех в окружении $ENV_NAME."
else
    echo "ошибка при установке зависимостей"
    exit 1
fi

echo "Для активации окружения -  conda activate $ENV_NAME"