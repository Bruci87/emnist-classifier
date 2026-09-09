import os
import joblib
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# -------------------------------------------------------------
# Processamento de Imagem Enviada
# -------------------------------------------------------------
def process_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None

    # Abre a imagem enviada em escala de cinza
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image, dtype=np.uint8)

    # Inverte se o fundo for branco e o desenho preto (EMNIST espera fundo preto)
    if np.mean(img_array) > 127:
        img_array = cv2.bitwise_not(img_array)

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

    # Alinhamento correto dos eixos para corresponder ao formato interno do EMNIST
    img_aligned = cv2.rotate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_aligned = cv2.flip(img_aligned, 0)

    # Normalização entre 0 e 1 e reshaping para (1, 784)
    return img_aligned.reshape(1, -1) / 255.0


# -------------------------------------------------------------
# Carregamento dos Modelos .pkl
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(cenario):
    model_path = os.path.join(BASE_DIR, "models", f"melhor_modelo_{cenario}.pkl")
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado em: {model_path}")
        return None
    return joblib.load(model_path)


# -------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------
st.set_page_config(page_title="Classificador EMNIST", layout="wide")
st.title("Classificador de Símbolos EMNIST")
st.write("Faça o upload de uma imagem do caractere para realizar a predição.")

tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    col_input, col_pred = st.columns([1, 1])
    
    with col_input:
        file_vf = st.file_uploader("Envie a imagem (PNG/JPG)", type=["png", "jpg", "jpeg"], key="file_vf")

    with col_pred:
        st.subheader("Resultado")
        if file_vf is not None:
            st.image(file_vf, width=150, caption="Imagem Enviada")
            img_processed = process_uploaded_image(file_vf)
            model = load_model("VF")
            if model and img_processed is not None:
                pred = model.predict(img_processed)[0]
                resultado_texto = "V" if str(pred).strip() in ["1", "V", "True", "1.0"] else "F"
                st.success(f"Resultado: **{resultado_texto}**")
        else:
            st.info("Aguardando arquivo de imagem...")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    col_input, col_pred = st.columns([1, 1])
    
    with col_input:
        file_num = st.file_uploader("Envie a imagem (PNG/JPG)", type=["png", "jpg", "jpeg"], key="file_num")

    with col_pred:
        st.subheader("Resultado")
        if file_num is not None:
            st.image(file_num, width=150, caption="Imagem Enviada")
            img_processed = process_uploaded_image(file_num)
            model = load_model("1_5")
            if model and img_processed is not None:
                pred = model.predict(img_processed)[0]
                st.success(f"Resultado: **{pred}**")
        else:
            st.info("Aguardando arquivo de imagem...")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    col_input, col_pred = st.columns([1, 1])
    
    with col_input:
        file_let = st.file_uploader("Envie a imagem (PNG/JPG)", type=["png", "jpg", "jpeg"], key="file_let")

    with col_pred:
        st.subheader("Resultado")
        if file_let is not None:
            st.image(file_let, width=150, caption="Imagem Enviada")
            img_processed = process_uploaded_image(file_let)
            model = load_model("A_E")
            if model and img_processed is not None:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Resultado: **{letra_pred}**")
        else:
            st.info("Aguardando arquivo de imagem...")
