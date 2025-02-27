"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib

import numpy as np
import cv2

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    win_size = [1024, 1024]
    step_size = [800, 800]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "lymph_custom"
    folder_name = "lymph_custom"
    save_root = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-10 AT Build Competent multi task DL model for tissue labeling\multi-segmodels\data\training_data\%s" % folder_name

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            # make sure to change jpg/png/tif
            "img": (".jpg", r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\0921 dataset\Split_0\Images"),
            "ann": (".npy", r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-10 AT Build Competent multi task DL model for tissue labeling\multi-segmodels\data\train\labels"),
        },
        "test": {
            # make sure to change jpg/png/tif
            "img": (".jpg", r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\dataset\Lymphocyte\0921 dataset\Split_0\Test\Images"),
            "ann": (".npy", r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-10 AT Build Competent multi task DL model for tissue labeling\multi-segmodels\data\test\labels"),
        },
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    # AT ignore valid for now - added break at end of loop
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        out_dir_im = "%s/%s/%s/%dx%d_%dx%d_%s/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
            "im"
        )
        # file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list = glob.glob(patterning("%s/*%s" % (img_dir, img_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)
        rm_n_mkdir(out_dir_im)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            print(file_path)
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                # save cropped images
                cv2.imwrite("{0}/{1}_{2:03d}.jpg".format(out_dir_im, base_name, idx), patch[:,:,:3])
                # only save last dim which is type map
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch[:,:,patch.shape[2]-1])
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()

