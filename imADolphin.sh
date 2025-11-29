#!/bin/bash

CSV_FILE="$1"

if [ -z "$CSV_FILE" ]; then
    echo "ERROR: Debes pasar un archivo .csv como argumento."
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: El archivo '$CSV_FILE' no existe."
    exit 1
fi

IMAGE_NAME="clasificacion_olivieri:latest"

# Verificar si la imagen existe
IMAGE_EXISTS=$(docker images -q "$IMAGE_NAME")
if [ -z "$IMAGE_EXISTS" ]; then
    echo "La imagen no existe. Creándola..."
    docker build -t "$IMAGE_NAME" .
else
    echo "La imagen $IMAGE_NAME ya existe."
fi

# Ejecutar contenedor pasándole el archivo al script Python
docker run \
    -v "$(pwd):/app" \
    "$IMAGE_NAME" \
    python inference.py "/app/$CSV_FILE"
