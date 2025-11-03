mensaje para el profe marin
Hola profe, para correr la app se necesita correr: 
 "pip install -r requirements.txt"
Dentro de cmd ya enfocado en la carpeta S21002378-Challenge\

Para revisar como se hizo el pipeline para el enmascarado revisar S21002378-Challenge4-Melanoma.ipynb

---

# Detector de Melanomas

Proyecto educativo que implementa una pequeña aplicación web para evaluar imágenes de lesiones cutáneas y devolver una predicción básica de riesgo (experimental, no clínica).

## Qué hace el código
- Permite subir una imagen de una lesión cutánea desde la interfaz web (`templates/index.html`).
- Aplica un pipeline de preprocesamiento (enmascarado/segmentación) para aislar la lesión y preparar la imagen.
- Ejecuta la inferencia con el modelo (o procedimiento de inferencia implementado en `app.py`) y devuelve una clasificación o puntuación de riesgo.
- Muestra el resultado en la interfaz y guarda temporalmente archivos en `files/` si es necesario.

## Cómo lo hace (resumen técnico)
- Preprocesamiento: el notebook `S21002378-Challenge4-Melanoma.ipynb` contiene los pasos experimentales para el enmascarado y la preparación de imágenes (lectura, segmentación, redimensionado y normalización).
- Modelo / inferencia: revisar `app.py` para ver cómo se carga el modelo (TensorFlow/PyTorch/u otro) y cómo se hace la predicción.
- Interfaz: `app.py` corre una app Flask que sirve `templates/index.html` para subir imágenes y mostrar resultados.

## Estructura del repositorio
- `app.py` — Aplicación Flask (servidor web para subir imágenes y ejecutar inferencia).
- `requirements.txt` — Dependencias de Python.
- `S21002378-Challenge4-Melanoma.ipynb` — Notebook con el pipeline de enmascarado y experimentos.
- `templates/index.html` — Interfaz web principal.
- `files/` — Carpeta para almacenar imágenes subidas o archivos temporales.
- `README.txt` — Este archivo (mensaje para el profe seguido del README refinado).

## Requisitos
- Python 3.8+ (recomendado).
- Instalar dependencias:

PowerShell (Windows):
```
pip install -r requirements.txt
```

O usando un entorno virtual:
```
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## Cómo ejecutar
1. Abrir PowerShell y situarse en la carpeta del proyecto (por ejemplo, `S21002378-Challenge4`).
2. Instalar dependencias si no están instaladas (`pip install -r requirements.txt`).
3. Ejecutar la aplicación:
```
python app.py
```
4. Abrir el navegador en la dirección que `app.py` indique (por ejemplo `http://127.0.0.1:5000`).

(Si `app.py` requiere variables de entorno o configuración adicional, revisa su cabecera y actualiza estas instrucciones en consecuencia.)

## Reproducir el pipeline de preprocesamiento
- Abrir `S21002378-Challenge4-Melanoma.ipynb` y ejecutar las celdas en orden. El notebook documenta el enmascarado y la preparación de imágenes.

## Limitaciones y advertencias
- Proyecto con fines educativos y experimentales — no usar para diagnóstico clínico.
- Resultados dependientes del modelo y datos; la robustez no está garantizada.
- Si los pesos o modelos no están incluidos, hay que añadir instrucciones o enlaces de descarga.

## Qué revisar o mejorar (sugerencias)
- Incluir el archivo de pesos o un enlace a su descarga.
- Añadir tests mínimos para la ruta de subida e inferencia.
- Registrar versiones concretas en `requirements.txt` para reproducibilidad.

---

Fin del README.