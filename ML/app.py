import os
import joblib
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# -------------------------------------------------------------
# Processamento de imagem do Canvas
# -------------------------------------------------------------
def process_canvas(canvas_result):
    """Extrai o desenho do canvas tratando transparência e fundo."""
    if canvas_result is None:
        return None

    try:
        raw_image = canvas_result.image_data
    except (RuntimeError, AttributeError):
        return None

    if raw_image is None:
        return None

    img_array = np.array(raw_image, dtype=np.uint8)

    if img_array.size == 0:
        return None

    # Se a imagem tiver 4 canais (RGBA), extraímos a escala de cinza combinando RGB ou Alpha
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_gray = np.max(img_array[:, :, :3], axis=2)
        if img_gray.max() == 0:
            img_gray = img_array[:, :, 3]
    elif len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    if img_gray.max() == 0:
        return None

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalização entre 0 e 1 e vetorização para 1x784
    return img_resized.reshape(1, -1) / 255.0


# -------------------------------------------------------------
# Carregamento do Modelo .pkl
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(cenario):
    """Carrega o modelo .pkl correspondente ao cenário."""
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
st.write("Desenhe o caractere no canvas e clique no botão correspondente para realizar a predição.")

tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    canvas_vf = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_vf"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Classificar V/F", key="btn_vf"):
            model = load_model("VF")
            if model:
                img_processed = process_canvas(canvas_vf)
                if img_processed is not None:
                    pred = model.predict(img_processed)[0]
                    res = "Verdadeiro (V)" if pred == 1 else "Falso (F)"
                    st.success(f"Resultado: {res}")
                else:
                    st.warning("Por favor, desenhe algo no canvas antes de classificar.")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    canvas_num = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_num"
    )
    
    if st.button("Classificar Dígito", key="btn_num"):
        model = load_model("1_5")
        if model:
            img_processed = process_canvas(canvas_num)
            if img_processed is not None:
                pred = model.predict(img_processed)[0]
                st.success(f"Dígito Predito: {pred}")
            else:
                st.warning("Por favor, desenhe algo no canvas antes de classificar.")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    canvas_let = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_let"
    )
    
    if st.button("Classificar Letra", key="btn_let"):
        model = load_model("A_E")
        if model:
            img_processed = process_canvas(canvas_let)
            if img_processed is not None:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Letra Predita: {letra_pred}")
            else:
                st.warning("Por favor, desenhe algo no canvas antes de classificar.")import os
import joblib
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# -------------------------------------------------------------
# Processamento de imagem do Canvas
# -------------------------------------------------------------
def process_canvas(canvas_result):
    """Extrai o desenho do canvas tratando transparência e fundo."""
    if canvas_result is None:
        return None

    try:
        raw_image = canvas_result.image_data
    except (RuntimeError, AttributeError):
        return None

    if raw_image is None:
        return None

    img_array = np.array(raw_image, dtype=np.uint8)

    if img_array.size == 0:
        return None

    # Se a imagem tiver 4 canais (RGBA), extraímos a escala de cinza combinando RGB ou Alpha
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Pega a intensidade máxima entre os canais RGB para detectar a tinta branca
        img_gray = np.max(img_array[:, :, :3], axis=2)
        
        # Se os canais RGB não pegarem nada, tenta o canal Alpha
        if img_gray.max() == 0:
            img_gray = img_array[:, :, 3]
    elif len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # Se mesmo assim tudo for 0, o usuário não desenhou nada
    if img_gray.max() == 0:
        return None

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalização entre 0 e 1 e vetorização para 1x784
    return img_resized.reshape(1, -1) / 255.0


# -------------------------------------------------------------
# Carregamento do Modelo .pkl
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(cenario):
    """Carrega o modelo .pkl correspondente ao cenário."""
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
st.write("Desenhe o caractere no canvas e clique no botão correspondente para realizar a predição.")

tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    canvas_vf = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        display_toolbar=True,  # Restaura a barra de ferramentas (desfazer, limpar, lixeira)
        key="canvas_vf"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Classificar V/F", key="btn_vf"):
            model = load_model("VF")
            if model:
                img_processed = process_canvas(canvas_vf)
                if img_processed is not None:
                    pred = model.predict(img_processed)[0]
                    res = "Verdadeiro (V)" if pred == 1 else "Falso (F)"
                    st.success(f"Resultado: {res}")
                else:
                    st.warning("Por favor, desenhe algo no canvas antes de classificar.")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    canvas_num = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        display_toolbar=True,  # Restaura a barra de ferramentas
        key="canvas_num"
    )
    
    if st.button("Classificar Dígito", key="btn_num"):
        model = load_model("1_5")
        if model:
            img_processed = process_canvas(canvas_num)
            if img_processed is not None:
                pred = model.predict(img_processed)[0]
                st.success(f"Dígito Predito: {pred}")
            else:
                st.warning("Por favor, desenhe algo no canvas antes de classificar.")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    canvas_let = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        display_toolbar=True,  # Restaura a barra de ferramentas
        key="canvas_let"
    )
    
    if st.button("Classificar Letra", key="btn_let"):
        model = load_model("A_E")
        if model:
            img_processed = process_canvas(canvas_let)
            if img_processed is not None:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Letra Predita: {letra_pred}")
            else:
                st.warning("Por favor, desenhe algo no canvas antes de classificar.")
