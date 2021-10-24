import csv
from os import listdir
from os.path import join

import cv2
from shapely.affinity import scale
from shapely.geometry import Polygon
from tqdm import tqdm

from pathlib import Path
xfact = 1.2
yfact = 1.8

input_img_dir = '/content/drive/MyDrive/colab/datasets/china_steel/public_training_data/public_training_data'
input_gt_csv = '/content/drive/MyDrive/colab/datasets/china_steel/public_training_data/public_training_data.csv'
output_dir = '/content/drive/MyDrive/colab/deep-text-recognition-benchmark/train'
output_gt_txt = join(output_dir, 'gt.txt')
output_img_dir = '/content/drive/MyDrive/colab/deep-text-recognition-benchmark/train/img'
Path(output_img_dir).mkdir(parents=True, exist_ok=True) 

with open(output_gt_txt, 'w') as output_gt_txt_f:
    with open(input_gt_csv, 'r') as input_gt_csv_f:
        reader = csv.DictReader(input_gt_csv_f)
        lines = list(reader)
        for d in tqdm(lines):
            filename = d['filename']
            label = d['label']

            poly = []
            poly += [d['top left x'], d['top left y']]
            poly += [d['top right x'], d['top right y']]
            poly += [d['bottom right x'], d['bottom right y']]
            poly += [d['bottom left x'], d['bottom left y']]

            poly = [float(x) for x in poly]
            poly = zip(poly[::2], poly[1::2])
            p = Polygon(poly)
            p = scale(p, xfact=xfact, yfact=yfact)
            bbox = p.bounds
            bbox = [int(x) for x in bbox]

            img_path = join(input_img_dir, f'{filename}.jpg')
            img = cv2.imread(img_path)
            cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            h, w, c = cropped.shape
            if w > 0:
                output_path = join(output_img_dir, f'{filename}.jpg')
                cv2.imwrite(output_path, cropped)

                output_gt_txt_f.write(f'{output_path}\t{label}\n')
