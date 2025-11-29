# Sistema de Inferencia - Predicción de Lluvia

Sistema Dockerizado para predecir lluvia basado en datos meteorológicos.

## Uso

```bash
./imADolphin.sh input.csv
```

El script automáticamente:
- Construye la imagen Docker si no existe
- Ejecuta la inferencia sobre el archivo CSV
- Genera `output.csv` con las predicciones

## Formato de Entrada

El CSV debe contener columnas meteorológicas: `Date`, `Location`, `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am`, `Temp3pm`, `RainToday`.

Ver [input.csv](input.csv) como ejemplo.

## Salida

El archivo `output.csv` contiene:
- **Predicción**: "Andá al parque tranquilo" o "Llevá paraguas."
- **Probabilidad de que no**: Probabilidad de NO lluvia (0-1)
- **Probabilidad de que sí**: Probabilidad de SÍ lluvia (0-1)
- **Confidence**: Nivel de confianza (0-1)

## Ejecución Manual

```bash
# Construir imagen
docker build -t clasificacion_olivieri:latest .

# Ejecutar inferencia
docker run -v $(pwd):/app clasificacion_olivieri:latest python inference.py /app/input.csv
```

## Solución de Problemas

Si `imADolphin.sh` no tiene permisos:
```bash
chmod +x imADolphin.sh
```
