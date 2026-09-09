import cv2
import numpy as np

def process_canvas(canvas_data):
    """Ajusta o desenho do canvas para o alinhamento exato do EMNIST treino."""
    if canvas_data is None:
        return None

    img_array = np.array(canvas_data)

    # Se o canvas estiver vazio (sem desenho)
    if img_array.max() == 0:
        return None

    # Extrai o canal de escala de cinza/alpha (se for RGBA)
    if len(img_array.shape) == 3:
        img_gray = img_array[:, :, 0]
    else:
        img_gray = img_array

    # 1. Redimensiona para 28x28
    img_resized = cv2.resize(img_gray.astype(np.uint8), (28, 28), interpolation=cv2.INTER_AREA)

    # 2. Normaliza os pixels entre 0 e 1 e transforma em vetor 1x784
    return img_resized.reshape(1, -1) / 255.0