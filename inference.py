
import joblib
import pandas as pd
import logging
import sys

from transformers import (
    RegionManagementTransformer,
    DateManagementTransformer,
    NullManagementTransformer,
    GroupedImputerTransformer,
    EncodingTransformer,
    NumericalScalerTransformer,
    KerasClassifierWrapper
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

def main():
    try:
        # 1. CARGAR PIPELINE
        logger.info("Cargando pipeline...")
        pipeline = joblib.load('pipeline.pkl')
        logger.info("Pipeline cargado")
        print("\n" + "="*60)

        # 2. LEER INPUT DESDE ARGUMENTO
        if len(sys.argv) < 2:
            logger.error("ERROR: No se recibió el archivo CSV como argumento.")
            sys.exit(1)
        input_csv = sys.argv[1]

        logger.info(f"Leyendo datos de entrada desde {input_csv}...")
        df_input = pd.read_csv(input_csv)
        logger.info(f"Datos cargados: {df_input.shape[0]} filas, {df_input.shape[1]} columnas")
        print("\n" + "="*60)

        # 3. HACER PREDICCIONES
        logger.info("Generando predicciones...")

        # Predicciones de clase (0 = No, 1 = Yes)
        predictions = pipeline.predict(df_input)

        # Probabilidades
        probabilities = pipeline.predict_proba(df_input)

        logger.info(f"{len(predictions)} predicciones generadas")

        # 4. PREPARAR OUTPUT
        logger.info("Preparando resultados...")

        # Convertir predicciones numéricas a labels
        prediction_labels = ['Andá al parque tranquilo' if pred == 0 else 'Llevá paraguas.' for pred in predictions]

        # Crear DataFrame de salida
        output_df = pd.DataFrame({
            'Predicción': prediction_labels,
            'Probabilidad de que no': probabilities[:, 0],
            'Probabilidad de que sí': probabilities[:, 1],
            'Confidence': probabilities.max(axis=1)
        })

        # 5. GUARDAR RESULTADOS
        output_path = './output.csv'
        output_df.to_csv(output_path, index=False)
        logger.info(f"Resultados guardados en {output_path}")

        # 6. RESUMEN DE RESULTADOS
        print("\n" + "="*60)
        print("RESUMEN DE PREDICCIONES")
        print("="*60)
        print(f"Total de predicciones: {len(predictions)}")
        print(f"  - No lloverá:  {(predictions == 0).sum()} ({(predictions == 0).sum()/len(predictions)*100:.1f}%)")
        print(f"  - Sí lloverá: {(predictions == 1).sum()} ({(predictions == 1).sum()/len(predictions)*100:.1f}%)")
        print(f"\nConfianza promedio: {output_df['Confidence'].mean():.2%}")

        print("Primeras líneas del archivo")
        print(output_df.head())

        print("\n" + "="*60)
        logger.info("Gracias, vuelva prontos.")
        print("="*60)

    except FileNotFoundError as e:
        logger.error(f"Error: Archivo no encontrado - {e}")
        logger.error("Asegúrate de montar el volumen con: docker run -v /ruta/local:/files ...")
        raise

    except Exception as e:
        logger.error(f"Error durante la inferencia: {e}")
        raise

if __name__ == "__main__":
    main()
