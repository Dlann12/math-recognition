"""
Image Preprocessing Pipeline for Handwritten Digit and Math Expression Recognition.

Implements grayscale conversion, background polarity inversion, bounding box cropping,
aspect-ratio preserved scaling, center-of-mass centering (MNIST standard),
and multi-character segmentation for math expressions.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def is_image_empty(img_array: np.ndarray, threshold: int = 20, min_active_pixels: int = 15) -> bool:
    """
    Check if an image is empty or has practically no drawing strokes.
    """
    if img_array is None or img_array.size == 0:
        return True

    # If image has multiple channels, convert to grayscale
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            # Check alpha channel first if transparent
            alpha = img_array[:, :, 3]
            if np.sum(alpha > 10) < min_active_pixels:
                return True
            gray = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Determine polarity: check if edges are mostly light
    h, w = gray.shape
    border_pixels = np.concatenate([
        gray[0, :], gray[h - 1, :], gray[:, 0], gray[:, w - 1]
    ])
    is_light_bg = np.median(border_pixels) > 127

    if is_light_bg:
        diff = 255 - gray
    else:
        diff = gray

    active_pixels = np.sum(diff > threshold)
    return bool(active_pixels < min_active_pixels)


def normalize_to_mnist_format(
    image: np.ndarray,
    target_size: Tuple[int, int] = (28, 28),
    box_size: int = 20
) -> Tuple[np.ndarray, bool]:
    """
    Preprocess a grayscale image of a single character into MNIST format:
    1. Polarity adjustment (strokes are white/bright, background is black/0).
    2. Bounding box cropping around the active stroke.
    3. Aspect-ratio preserving resize into box_size x box_size (e.g. 20x20).
    4. Centering inside target_size x target_size (e.g. 28x28) using center of mass.
    5. Normalizing pixel values to [0.0, 1.0].

    Returns:
        (processed_img, is_valid):
            processed_img is of shape (1, 28, 28, 1) float32
            is_valid is False if image has insufficient stroke pixels
    """
    if image is None or image.size == 0:
        return np.zeros((1, target_size[0], target_size[1], 1), dtype=np.float32), False

    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # Handle alpha channel: if background is transparent, composite onto white or extract alpha
            alpha = image[:, :, 3]
            rgb = image[:, :, :3]
            if np.max(alpha) > 0 and np.min(alpha) == 0:
                # Transparent canvas: strokes are where alpha > 0
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                # Where alpha is 0, make it white background
                gray[alpha == 0] = 255
            else:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Determine background polarity: check outer borders
    h, w = gray.shape
    border_pixels = np.concatenate([
        gray[0, :], gray[h - 1, :], gray[:, 0], gray[:, w - 1]
    ])
    is_light_bg = np.median(border_pixels) > 127

    if is_light_bg:
        binary_img = 255 - gray
    else:
        binary_img = gray.copy()

    # Mild thresholding to eliminate compression/canvas noise
    _, thresh = cv2.threshold(binary_img, 30, 255, cv2.THRESH_TOZERO)

    # Find bounding box of non-zero pixels
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 15:
        return np.zeros((1, target_size[0], target_size[1], 1), dtype=np.float32), False

    x, y, bw, bh = cv2.boundingRect(coords)

    # Crop the character region
    cropped = thresh[y:y + bh, x:x + bw]

    # Resize keeping aspect ratio so that max dimension equals box_size
    if bh > bw:
        scale = box_size / float(bh)
        new_h = box_size
        new_w = max(1, int(round(bw * scale)))
    else:
        scale = box_size / float(bw)
        new_w = box_size
        new_h = max(1, int(round(bh * scale)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place resized character into a 28x28 empty frame centered by bounding box
    padded = np.zeros(target_size, dtype=np.float32)
    start_y = (target_size[0] - new_h) // 2
    start_x = (target_size[1] - new_w) // 2
    padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    # Center by Center of Mass (standard MNIST alignment)
    cy, cx = ndimage.center_of_mass(padded)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = int(round(target_size[0] / 2.0 - cy))
        shift_x = int(round(target_size[1] / 2.0 - cx))
        # Clamp shifts to prevent clipping outside the frame
        shift_y = max(min(shift_y, 4), -4)
        shift_x = max(min(shift_x, 4), -4)
        padded = ndimage.shift(padded, (shift_y, shift_x), cval=0.0)

    # Clip and normalize to [0.0, 1.0]
    padded = np.clip(padded, 0.0, 255.0) / 255.0

    # Expand dims to match (1, 28, 28, 1)
    processed = padded.reshape(1, target_size[0], target_size[1], 1).astype(np.float32)
    return processed, True


def preprocess_single_image(image_input) -> Tuple[Optional[np.ndarray], bool, str]:
    """
    Takes an input image (PIL Image, numpy array, or file upload) and processes it.
    Returns:
        (processed_array, is_valid, message)
    """
    if image_input is None:
        return None, False, "Tidak ada gambar yang diberikan."

    if isinstance(image_input, Image.Image):
        img_arr = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        img_arr = image_input
    else:
        try:
            pil_img = Image.open(image_input)
            img_arr = np.array(pil_img)
        except Exception as e:
            return None, False, f"Format gambar tidak didukung: {str(e)}"

    if is_image_empty(img_arr):
        return None, False, "Kanvas atau gambar kosong. Silakan gambar atau unggah angka terlebih dahulu."

    processed, is_valid = normalize_to_mnist_format(img_arr)
    if not is_valid:
        return None, False, "Garis gambar terlalu tipis atau tidak terdeteksi. Silakan coba lagi."

    return processed, True, "Berhasil diproses."


def segment_math_expression(image_input) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment a handwritten mathematical expression into individual characters.
    Handles multi-stroke characters (e.g. '=' with 2 horizontal bars, '÷' with dots, '+' with 2 strokes).

    Returns:
        list of tuples: [(preprocessed_char_tensor, (x, y, w, h)), ...]
        sorted from left to right.
    """
    if isinstance(image_input, Image.Image):
        img_arr = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        img_arr = image_input
    else:
        pil_img = Image.open(image_input)
        img_arr = np.array(pil_img)

    if is_image_empty(img_arr):
        return []

    # Convert to grayscale
    if len(img_arr.shape) == 3:
        if img_arr.shape[2] == 4:
            alpha = img_arr[:, :, 3]
            rgb = img_arr[:, :, :3]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray[alpha == 0] = 255
        else:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_arr.copy()

    # Determine polarity
    h, w = gray.shape
    border_pixels = np.concatenate([
        gray[0, :], gray[h - 1, :], gray[:, 0], gray[:, w - 1]
    ])
    is_light_bg = np.median(border_pixels) > 127

    if is_light_bg:
        binary_img = 255 - gray
    else:
        binary_img = gray.copy()

    # Thresholding
    _, thresh = cv2.threshold(binary_img, 30, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        # Filter very small noise specks (< 4px)
        if bw >= 4 and bh >= 4 and (bw * bh >= 20):
            raw_boxes.append((bx, by, bw, bh))

    if not raw_boxes:
        return []

    # Sort boxes primarily by X coordinate
    raw_boxes = sorted(raw_boxes, key=lambda b: b[0])

    # Merge vertically stacked or overlapping components (like = or ÷ or disconnected strokes)
    merged_boxes = []
    for b in raw_boxes:
        bx, by, bw, bh = b
        merged = False
        for i, mb in enumerate(merged_boxes):
            mx, my, mw, mh = mb
            # Check horizontal overlap
            x_overlap = max(0, min(bx + bw, mx + mw) - max(bx, mx))
            # Minimum width between the two
            min_w = min(bw, mw)

            # If there is significant horizontal overlap and vertical proximity (e.g. '=')
            if (min_w > 0 and (x_overlap / min_w) > 0.45) or (abs(bx - mx) < 15 and abs((bx + bw) - (mx + mw)) < 15):
                # Calculate vertical gap
                vertical_gap = max(0, max(by, my) - min(by + bh, my + mh))
                if vertical_gap < max(bh, mh) * 1.5:
                    new_x = min(bx, mx)
                    new_y = min(by, my)
                    new_w = max(bx + bw, mx + mw) - new_x
                    new_h = max(by + bh, my + mh) - new_y
                    merged_boxes[i] = (new_x, new_y, new_w, new_h)
                    merged = True
                    break

        if not merged:
            merged_boxes.append(b)

    # Sort final merged boxes from left to right
    merged_boxes = sorted(merged_boxes, key=lambda b: b[0])

    segmented_characters = []
    for bx, by, bw, bh in merged_boxes:
        # Add a small padding margin around the character
        pad = 4
        py1 = max(0, by - pad)
        py2 = min(h, by + bh + pad)
        px1 = max(0, bx - pad)
        px2 = min(w, bx + bw + pad)

        char_crop = binary_img[py1:py2, px1:px2]
        char_tensor, is_valid = normalize_to_mnist_format(char_crop)
        if is_valid:
            segmented_characters.append((char_tensor, (bx, by, bw, bh)))

    return segmented_characters
