import numpy as np
from .misc.utils import get_bounding_box



overlay = np.copy((input_image).astype(np.uint8))

inst_list = list(np.unique(inst_map))  # get list of instances
inst_list.remove(0)  # remove background

inst_rng_colors = random_colors(len(inst_list))
inst_rng_colors = np.array(inst_rng_colors) * 255
inst_rng_colors = inst_rng_colors.astype(np.uint8)

for inst_idx, inst_id in enumerate(inst_list):
    inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
    y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
    y1 = y1 - 2 if y1 - 2 >= 0 else y1
    x1 = x1 - 2 if x1 - 2 >= 0 else x1
    x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
    y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
    inst_map_crop = inst_map_mask[y1:y2, x1:x2]
    contours_crop = cv2.findContours(
        inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # only has 1 instance per map, no need to check #contour detected by opencv
    contours_crop = np.squeeze(
        contours_crop[0][0].astype("int32")
    )  # * opencv protocol format may break
    contours_crop += np.asarray([[x1, y1]])  # index correction
    if type_map is not None:
        type_map_crop = type_map[y1:y2, x1:x2]
        type_id = np.unique(type_map_crop).max()  # non-zero
        inst_colour = type_colour[type_id]
    else:
        inst_colour = (inst_rng_colors[inst_idx]).tolist()
    cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)