import cv2
import numpy as np

def Mosaic(img, bboxes, face_ids):
    """
    Apply Gaussian blur on detected faces except those recognized as known.
    """
    output_img = img.copy()
    for bbox, face_id in zip(bboxes, face_ids):
        if face_id == 'unknown':
            x0, y0, x1, y1 = map(int, bbox)
            roi = img[y0:y1, x0:x1]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            output_img[y0:y1, x0:x1] = blurred_roi
    return output_img

def Get_normal_bbox(size, bboxes):
    """
    Normalize bounding boxes to ensure they stay within image bounds.
    """
    new_bboxes = []
    for bbox in bboxes:
        x_min = max(0, bbox[0])
        y_min = max(0, bbox[1])
        x_max = min(size[1], bbox[2])
        y_max = min(size[0], bbox[3])
        if x_max > x_min and y_max > y_min:
            new_bboxes.append([x_min, y_min, x_max, y_max])
    return new_bboxes