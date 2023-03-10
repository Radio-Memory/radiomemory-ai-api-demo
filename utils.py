import base64
import io

import cv2
import numpy as np
from PIL import Image


def process_ret_api(json_response, h, w, anomaly2see):
    pan_out = np.zeros(shape=(h, w), dtype=np.uint8)
    tmp_pan_out = np.zeros(shape=(h, w), dtype=np.uint8)

    for tooth in json_response:
        if not len(tooth["anomalies"]):
            continue

        for anomaly in tooth["anomalies"]:
            anomaly_name = anomaly["anomaly_name"]

            if anomaly_name != anomaly2see:
                continue

            decoded_heatmap = base64.b64decode(anomaly["cropped_heatmap"])
            decoded_heatmap = Image.open(io.BytesIO(decoded_heatmap))

            bbox_xywh = anomaly["bbox_xywh"]
            min_x = bbox_xywh[0]
            min_y = bbox_xywh[1]
            max_x = bbox_xywh[0] + bbox_xywh[2]
            max_y = bbox_xywh[1] + bbox_xywh[3]

            tmp_pan_out[min_y:max_y, min_x:max_x] = decoded_heatmap

            pan_out = cv2.addWeighted(pan_out, 1.0, tmp_pan_out, 1.0, 0)

    color = (255, 0, 0)
    heatmap = pan_out.astype(np.float64) / 255
    heatmap_image = np.stack(
        [heatmap * color[0], heatmap * color[1], heatmap * color[2]], axis=2
    ).astype(np.uint8)

    return heatmap_image
