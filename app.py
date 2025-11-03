# =============================================================================
# APLICACIÓN WEB DE ANÁLISIS DE PIEL (app.py - V2.0 FINAL)
# =============================================================================
import cv2
import numpy as np
import pandas as pd
import os
import io
import base64
import matplotlib
matplotlib.use('Agg') # Usar un backend no interactivo para el servidor
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURACIÓN INICIAL DE LA APP Y EL MODELO ---
app = Flask(__name__)
model = None
scaler = None
df_train_global = None # DataFrame global para usar en el ploteo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, 'files')

feature_names = [
    'irregularidad_std_norm', 'color_r_promedio', 'color_g_promedio',
    'iluminacion_promedio', 'forma_area', 'forma_perimetro',
    'forma_circularidad', 'forma_solidez', 'forma_hu_moment_0', 'forma_hu_moment_1'
]
best_features_to_use = ['irregularidad_std_norm', 'color_r_promedio', 'color_g_promedio']

# --- PIPELINE DE PROCESAMIENTO DE IMAGEN (AHORA DEVUELVE PASOS) ---
def procesar_y_obtener_pasos(img_color):
    visual_steps = {} # Diccionario para guardar las imágenes de los pasos

    # 1. Corrección de Iluminación
    alto, ancho, _ = img_color.shape
    kernel_size = int(ancho * 0.25)
    if kernel_size % 2 == 0: kernel_size += 1
    fondo_estimado = cv2.GaussianBlur(img_color, (kernel_size, kernel_size), 0)
    img_corregida = cv2.divide(img_color, fondo_estimado, scale=255)
    visual_steps['Iluminacion Corregida'] = img_corregida

    # 2. Segmentación con Otsu
    gris_corregido = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)
    _, binarizada_otsu = cv2.threshold(gris_corregido, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Limpieza Morfológica (Apertura)
    kernel_limpieza = np.ones((3, 3), np.uint8)
    mascara_limpia = cv2.morphologyEx(binarizada_otsu, cv2.MORPH_OPEN, kernel_limpieza, iterations=2)
    visual_steps['Mascara Limpia'] = mascara_limpia

    # 4. Post-procesamiento Final
    mascara_solo_lunar = np.zeros_like(mascara_limpia)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_limpia, 4, cv2.CV_32S)
    if num_labels > 1:
        idx_max_area = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mascara_solo_lunar[labels == idx_max_area] = 255
        
    kernel_cierre = np.ones((15, 15), np.uint8)
    mascara_final = cv2.morphologyEx(mascara_solo_lunar, cv2.MORPH_CLOSE, kernel_cierre, iterations=2)
    visual_steps['Mascara Final'] = mascara_final

    # 5. Dibujar contorno para visualización
    contornos_vis = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    contornos, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        cv2.drawContours(contornos_vis, contornos, -1, (255, 0, 0), 2)
    visual_steps['Contorno Final'] = contornos_vis
    
    return mascara_final, visual_steps

def calcular_features(imagen_color, mascara):
    # (Esta función no cambia)
    features = {}
    if np.sum(mascara) == 0: return {fname: 0 for fname in feature_names}
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos: return {fname: 0 for fname in feature_names}
    cnt = max(contornos, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        centroide_x, centroide_y = M["m10"] / M["m00"], M["m01"] / M["m00"]
        distancias = np.sqrt(np.sum((cnt.squeeze() - [centroide_x, centroide_y])**2, axis=1))
        media_dist = distancias.mean()
        features['irregularidad_std_norm'] = np.std(distancias / media_dist) if media_dist > 0 else 0
    else: features['irregularidad_std_norm'] = 0
    media_bgr = cv2.mean(imagen_color, mask=mascara)
    features['color_r_promedio'], features['color_g_promedio'], _ = media_bgr[2], media_bgr[1], media_bgr[0]
    gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
    features['iluminacion_promedio'] = cv2.mean(gris, mask=mascara)[0]
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    features['forma_area'], features['forma_perimetro'] = area, perimetro
    features['forma_circularidad'] = (4 * np.pi * area) / (perimetro**2) if perimetro > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    features['forma_solidez'] = float(area) / hull_area if hull_area > 0 else 0
    momentos = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(momentos)
    features['forma_hu_moment_0'], features['forma_hu_moment_1'] = hu_moments[0][0], hu_moments[1][0]
    return features

def train_best_model():
    global model, scaler, df_train_global
    print("INFO: Entrenando el modelo con los datos base...")
    rutas_entrenamiento = [os.path.join(FILES_DIR, f'mela{i}.jpeg') for i in range(1, 5)]
    lista_features_train = []
    for ruta in rutas_entrenamiento:
        img = cv2.imread(ruta)
        if img is None: continue
        benigno_img, maligno_img = img[57:, :104], img[57:, 110:]
        mascara_b, _ = procesar_y_obtener_pasos(benigno_img)
        features_b = calcular_features(benigno_img, mascara_b); features_b['clase'] = 'Benigno'
        lista_features_train.append(features_b)
        mascara_m, _ = procesar_y_obtener_pasos(maligno_img)
        features_m = calcular_features(maligno_img, mascara_m); features_m['clase'] = 'Maligno'
        lista_features_train.append(features_m)

    df_train_global = pd.DataFrame(lista_features_train)
    X_train = df_train_global[best_features_to_use]
    y_train = LabelEncoder().fit_transform(df_train_global['clase'])

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model = RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train)
    print("INFO: Modelo entrenado y listo.")

def encode_image_for_html(img_np):
    """Convierte una imagen de OpenCV a un string Base64 para HTML."""
    _, buffer = cv2.imencode('.jpg', img_np)
    return base64.b64encode(buffer).decode('utf-8')

# --- RUTAS DE LA APLICACIÓN WEB ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file_bytes = np.fromfile(request.files['image'], np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 1. Procesar la imagen y obtener los pasos visuales
    mascara, visual_steps = procesar_y_obtener_pasos(img)
    
    # 2. Convertir los pasos a Base64 para enviarlos al HTML
    encoded_steps = {name: encode_image_for_html(step_img) for name, step_img in visual_steps.items()}
    
    # 3. Calcular características y clasificar
    features_dict = calcular_features(img, mascara)
    df_new = pd.DataFrame([features_dict])[best_features_to_use]
    df_new_scaled = scaler.transform(df_new)
    probabilidades = model.predict_proba(df_new_scaled)[0]
    prob_maligno = probabilidades[1]
    porcentaje = prob_maligno * 100
    categoria = "Benigno" if porcentaje <= 40 else "Inconcluso" if 41 <= 70 else "Probablemente Maligno"

    # 4. Generar el gráfico de dispersión
    feat1, feat2 = best_features_to_use[0], best_features_to_use[1] # Usar las 2 primeras features más importantes
    fig, ax = plt.subplots(figsize=(6, 4))
    df_benignos = df_train_global[df_train_global['clase'] == 'Benigno']
    df_malignos = df_train_global[df_train_global['clase'] == 'Maligno']
    ax.scatter(df_benignos[feat1], df_benignos[feat2], c='blue', label='Benigno', alpha=0.7)
    ax.scatter(df_malignos[feat1], df_malignos[feat2], c='red', label='Maligno', alpha=0.7)
    ax.scatter(df_new[feat1], df_new[feat2], c='magenta', label='Tu Imagen', s=150, marker='X', edgecolors='k')
    ax.set_xlabel(feat1.replace('_', ' ').title())
    ax.set_ylabel(feat2.replace('_', ' ').title())
    ax.set_title('Tu Imagen vs. Datos de Entrenamiento')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({
        'category': categoria,
        'probability': f'{porcentaje:.2f}',
        'steps': encoded_steps,
        'plot': plot_base64
    })

# --- INICIO DE LA APLICACIÓN ---
if __name__ == '__main__':
    train_best_model()
    app.run(host='0.0.0.0', port=5000, debug=False)