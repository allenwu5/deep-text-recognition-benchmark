import argparse
import csv
from os.path import exists, join

import cv2
from tqdm import tqdm


def main(opt):
    with open(opt.dataset_gt_path, 'w') as gt_file:
        for csv_path in [opt.train_gt_csv, opt.train2_gt_csv]:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                lines = list(reader)
                for d in tqdm(lines):
                    filename = d['filename']
                    label = d['label']
                    img_path = join(opt.img_dir, f'{filename}.jpg')

                    if exists(img_path):
                        gt_file.write(f'{img_path}\t{label}\n')

                        # Data augmentation
                        img = cv2.imread(img_path)

                        width = int(img.shape[1] * 60 / 100)
                        height = int(img.shape[0] * 70 / 100)
                        dim = (width, height)
                        resized = cv2.resize(
                            img, dim, interpolation=cv2.INTER_AREA)
                        # img = cv2.rotate(img, cv2.ROTATE_180)

                        img_path = join(opt.img_dir, f'{filename}_aug.jpg')
                        cv2.imwrite(img_path, resized)
                        gt_file.write(f'{img_path}\t{label}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='')
    parser.add_argument('--dataset_gt_path', help='')
    parser.add_argument('--train_gt_csv', help='')
    parser.add_argument('--train2_gt_csv', help='')
    opt = parser.parse_args()
    main(opt)
