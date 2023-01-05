import cv2
import numpy as np


def extract_roi(img, resize=(256, 256)):
    roi = crop_roi(img)
    img_normalized = truncation_normalization(roi)
    img_resized = cv2.resize(img_normalized, resize)
    final_img = np.stack((img_resized,) * 3, axis=-1)
    return final_img


def crop_roi(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("uint8")
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    img_cropped = img[y:y + h, x:x + w]
    return img_cropped


def truncation_normalization(img):
    Pmin = np.percentile(img[img != 0], 5)
    Pmax = np.percentile(img[img != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)

    if Pmax == Pmin:
        normalized = truncated
    else:
        normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[img == 0] = 0
    return normalized
