import csv
from os.path import exists, join

import cv2
from tqdm import tqdm

img_dir = '/content/rects'

train_gt_csv = '/content/drive/MyDrive/2021_china_steel_ocr/label/public_training_data.csv'
train_test_gt_csv = '/content/drive/MyDrive/2021_china_steel_ocr/label/public_testing_data.csv'

output_gt_txt = '/content/gt.txt'


with open(output_gt_txt, 'w') as gt_file:
    for csv_path in [train_test_gt_csv, train_gt_csv]:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            lines = list(reader)
            for d in tqdm(lines):
                filename = d['filename']
                label = d['label']
                img_path = join(img_dir, f'{filename}.jpg')

                if exists(img_path):
                    gt_file.write(f'{img_path}\t{label}\n')

                    img = cv2.imread(img_path)

                    width = int(img.shape[1] * 60 / 100)
                    height = int(img.shape[0] * 70 / 100)
                    dim = (width, height)
                    resized = cv2.resize(
                        img, dim, interpolation=cv2.INTER_AREA)
                    # img = cv2.rotate(img, cv2.ROTATE_180)

                    img_path = join(img_dir, f'{filename}_resize_70.jpg')
                    cv2.imwrite(img_path, resized)
                    gt_file.write(f'{img_path}\t{label}\n')
