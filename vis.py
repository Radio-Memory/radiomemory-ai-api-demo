from collections import defaultdict
from typing import Tuple, Union

import cv2
import distinctipy
import matplotlib.pylab as plt
import numpy as np
from PIL import Image


def preprocess_image_draw(image: Union[Image.Image, np.ndarray]):

    image = np.array(image)  # Convert if PIL, copy if numpy
    if len(image.shape) == 2:  # GRAYSCALE
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        pass
    else:
        raise ValueError("Unsupported number of dims on image.")

    return image


def draw_tooth(
    image: np.ndarray,
    pt0: list,
    pt1: list,
    tooth_name: str,
    color: tuple,
    size_factor=1.0,
    draw_axis=False,
):

    x1, y1 = pt0
    height, width = image.shape[:2]
    text = f"{tooth_name}"
    draw_scale = max(height, width) / 2000
    x2, y2 = pt1
    # x2, y2 = x2 * width, y2 * height
    px = min(x1, x2) + np.abs(x1 - x2) // 2
    py = min(y1, y2) + np.abs(y1 - y2) // 2
    if draw_axis:
        image = cv2.line(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            int(2 * draw_scale),
        )
    cv2.putText(
        image,
        text,
        (int(px - 20 * draw_scale / 2000), int(py)),
        cv2.FONT_HERSHEY_SIMPLEX,
        size_factor * draw_scale,
        color,
        int(size_factor * draw_scale * 3),
    )

    return image


def draw_longaxis_output(
    image: Union[Image.Image, np.ndarray],
    keypoints: list,
    color: Tuple = (0, 255, 0),
    th: float = 0.14,
    size_factor=1.0,
    draw_axis=False,
):

    image = preprocess_image_draw(image)
    teeth_map = defaultdict(list)
    for keypoint in keypoints:
        tooth_name = keypoint["class_name"].split("_")[0]
        teeth_map[tooth_name].append(keypoint)
    for tooth_name, keypoints in teeth_map.items():
        if np.mean([p["score"] for p in keypoints]) < th:
            continue
        pt0 = keypoints[0]["point"]
        pt1 = keypoints[1]["point"]
        image = draw_tooth(
            image,
            pt0,
            pt1,
            tooth_name,
            color,
            size_factor,
            draw_axis=draw_axis,
        )

    return image


def draw_panorogram(image, contours_pairs, closed=False):
    dimage = preprocess_image_draw(image)
    alpha = 0.5
    COLOR_MAP = {
        "ContMand": (0, 255, 255),
        "CanManDir": (255, 0, 0),
        "CanManEsq": (255, 0, 0),
        "RebAlvInf": (0, 255, 0),
        "RebAlvSup": (0, 255, 0),
        "SeioMaxDir": (0, 0, 255),
        "SeioMaxEsq": (0, 0, 255),
        "FossaNasal": (255, 255, 0),
    }

    for pair in contours_pairs:
        if "CanMan" in pair[0]:
            overlay = np.zeros(shape=dimage.shape, dtype=np.uint8)
            overlay = cv2.drawContours(
                overlay,
                [np.array(pair[1]).astype(int)],
                -1,
                color=COLOR_MAP[pair[0]],
                thickness=-1,
            )
        else:
            overlay = np.zeros(shape=dimage.shape, dtype=np.uint8)
            overlay = cv2.polylines(
                overlay,
                [np.array(pair[1]).astype(int)],
                isClosed=closed,
                color=COLOR_MAP[pair[0]],
                thickness=int(max(dimage.shape) / 500),
            )
        dimage = cv2.addWeighted(overlay, alpha, dimage, 1, 0)

    return dimage


def draw_bbox(image, coords, color=(225, 0, 0), text=None, text_below=False):
    """Draw bbox on image, expect an int image"""
    dimage = image
    height, width = dimage.shape[:2]
    min_dim = min(height, width)
    x1, y1, x2, y2 = coords

    thickness = 1 + int(min_dim / 600)
    dimage = cv2.rectangle(
        dimage, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness
    )
    if text is None:
        return dimage

    if not text_below:
        x_text, y_text = int(x1), int(y1 - min_dim / 100)
    else:
        x_text, y_text = int(x1), int(y1 + min_dim / 100)

    dimage = cv2.putText(
        dimage,
        text,
        (int(x_text), int(y_text)),
        cv2.FONT_HERSHEY_SIMPLEX,
        min_dim / 1000,
        color,
        int(0.7 * thickness),
        cv2.LINE_AA,
    )

    return dimage


def draw_bboxes(image, bboxes, th=0.5):

    dimage = preprocess_image_draw(image)
    for idx, bbox in enumerate(bboxes):
        if bbox["score"] < th:
            continue
        color = plt.get_cmap("hsv")(idx / 32)  # Number of teeth
        color = [int(x * 255) for x in color]
        text = f"{bbox['class_name']} {bbox['score']:.2f}"
        dimage = draw_bbox(
            dimage,
            bbox["bbox"],
            color=color,
            text=text,
        )

    return dimage


def contour2mask(contours, w, h):

    conv_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    conv_mask = cv2.fillPoly(
        conv_mask,
        [np.array(tcont, dtype=int).reshape((-1, 1, 2)) for tcont in contours],
        color=255,
    )
    return conv_mask


def draw_masks(image, masks):
    dimage = preprocess_image_draw(image)
    alpha = 0.2
    for idx, mask in enumerate(masks):
        mask = mask / 255
        color = plt.get_cmap("hsv")(idx / len(masks))
        mask_color = (
            255 * np.stack([color[0] * mask, color[1] * mask, color[2] * mask], axis=2)
        ).astype(np.uint8)
        dimage = cv2.addWeighted(dimage, 1, mask_color, 1 - alpha, 0)
    return dimage


def draw_heatmap(image, heatmap):
    dimage = preprocess_image_draw(image)
    alpha = 0.2
    dimage = cv2.addWeighted(dimage, 1, heatmap, 1 - alpha, 0)
    return dimage


def draw_contours(image, contours, color=(255, 0, 0), closed=False):

    dimage = preprocess_image_draw(image)

    dimage = cv2.polylines(
        dimage.copy(),
        [np.array(cont).astype(int) for cont in contours],
        isClosed=closed,
        color=color,
        thickness=2,
    )
    return dimage


def draw_procedures_output(
    img,
    entities,
    point_names=None,
    plot_labels=False,
):
    img = preprocess_image_draw(img)

    height, width = img.shape[:2]
    scale = max(img.shape) / 2000
    shown_cls = []

    max_upper_limit = height
    mand_lower_limit = 0

    tooth_map = defaultdict(list)
    for e in entities:
        tooth_map[e["tooth"]].append(e)

    CLASSES = sorted(list({e["class_name"] for e in entities}))

    colors = distinctipy.get_colors(len(CLASSES), rng=139)

    for e in entities:
        point = e["line"]
        if point[0][1] < point[1][1]:  # Mandibula
            if point[1][1] > mand_lower_limit:
                mand_lower_limit = point[1][1]
        if point[0][1] > point[1][1]:  # Maxila
            if point[1][1] < max_upper_limit:
                max_upper_limit = point[1][1]

    for i, (tooth, ents) in enumerate(tooth_map.items()):
        point = ents[0]["line"]
        label = ents[0]["class_name"]
        if point[0][1] < point[1][1]:  # Mandibula

            xb, yb = point[1][0], point[1][1]  # botton coods
            xt, yt = point[0][0], point[0][1]  # top coords

            ax, ay = xb - xt, yb - yt  # center point
            pv = np.array([0, ay]) / 6

            for j, e in enumerate(ents):
                color = colors[CLASSES.index(e["class_name"])]
                color = [x * 255 for x in color]
                offset = j
                shown_cls.append(e["class_name"])
                img = cv2.circle(
                    img,
                    (
                        int(xb + offset * pv[0]),
                        int(1.05 * mand_lower_limit + offset * pv[1]),
                    ),
                    int(max(img.shape) / 200),
                    color,
                    -1,
                )
        elif point[0][1] > point[1][1]:  # Maxila

            xb, yb = point[0][0], point[0][1]
            xt, yt = point[1][0], point[1][1]

            ax, ay = xt - xb, yt - yb

            pv = np.array([0, ay]) / 6

            for j, e in enumerate(ents):
                # color = plt.get_cmap("hsv")(
                #     CLASSES.index(l) / len(CLASSES)
                # )  # Number of classes on COCO
                color = colors[CLASSES.index(e["class_name"])]
                color = [x * 255 for x in color]
                # offset = j - len(label) // 2
                offset = j
                shown_cls.append(e["class_name"])
                img = cv2.circle(
                    img,
                    (
                        int(xt + offset * pv[0]),
                        int(0.95 * max_upper_limit + offset * pv[1]),
                    ),
                    int(max(img.shape) / 200),
                    color,
                    -1,
                )

    bimg = np.zeros((height, int(width + 500 * scale), 3), dtype=np.uint8)
    bimg[:height, :width, :] = img
    width = bimg.shape[1]
    for i, _cls in enumerate(list(set(shown_cls))):
        # color = plt.get_cmap("hsv")(
        #     CLASSES.index(_cls) / len(CLASSES)
        # )  # Number of classes on COCO
        color = colors[CLASSES.index(_cls)]
        color = [x * 255 for x in color]
        text = _cls
        font_scale = scale * 1
        img = cv2.putText(
            bimg,
            text,
            (
                int(width - 500 * scale + scale * 50),
                int(height - 500 * scale + i * 40 * scale),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness=int(3 * scale),
        )

    return img


def draw_points(image, entities):
    dimage = preprocess_image_draw(image)
    width, height = image.size
    draw_scale = max(height, width) / 2000
    for ent in entities:
        x, y = ent["point"]
        dimage = cv2.circle(dimage, (int(x), int(y)), int(10 * draw_scale), (0, 255, 0), -1)
        dimage = cv2.putText(
            dimage,
            ent["class_name"],
            (int(x), int(y - width / 80)),
            cv2.FONT_HERSHEY_SIMPLEX,
            draw_scale,
            (255, 0, 0),
            int(4 * draw_scale),
            cv2.LINE_AA,
        )

    return dimage
