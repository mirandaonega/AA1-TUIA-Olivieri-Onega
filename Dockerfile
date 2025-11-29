FROM python:3.11-slim

# Configurar directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt ./

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de archivos
COPY . .

# Verificar que el pipeline existe
RUN test -f pipeline.pkl || (echo "ERROR: pipeline.pkl no encontrado" && exit 1)

# Comando por defecto
CMD ["python", "./inference.py"]
