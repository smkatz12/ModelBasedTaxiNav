import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import os

dataset_dir = '/scratch/smkatz/class/CS231A/E16Data_omstraj/'
outfile = 'downsampled_64.h5'

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
strideh = 9
stridew = 30
numPix = 32
width = 1920//stridew
height = 576//strideh

def get_data(dataset_dir, verbose=True):
    # if verbose:
    #     print("\nReading " + csv_path)

    labels = pd.read_csv(dataset_dir + 'labels.csv')
    # y = np.array((data.distance_to_centerline_meters, data.heading_error_degrees,
    #               data.downtrack_position_meters)).T.astype('float32')
    X = np.zeros([len(labels), height, width]).astype('float32')
    y = np.zeros([len(labels), 7]).astype('float32')

    for i, fn in enumerate(labels.filename):
        img = np.array(cv2.cvtColor(cv2.imread(
            dataset_dir + fn), cv2.COLOR_BGRA2BGR)[:, :, ::-1])
        data = labels.loc[labels['filename'] == fn].iloc[0]
        right_labels = [data['rightx1'], data['righty1'],
                        data['rightx2'], data['righty2']]
        left_labels = [data['leftx1'], data['lefty1'],
                    data['leftx2'], data['lefty2']]
        img_ds, right_line, left_line = downsample_image(img, right_labels, left_labels)

        # rho_right, phi_right = to_angle_offset(right_line)
        # rho_left, phi_left = to_angle_offset(left_line)

        # m_right, b_right = -right_line[:2] / right_line[2]
        # m_left, b_left = -left_line[:2] / left_line[2]
        ar, br, cr = right_line
        m_right = -ar / br
        b_right = -cr / br
        al, bl, cl = left_line
        m_left = -al / bl
        b_left = -cl / bl

        X[i] = img_ds
        y[i] = np.array([m_right, b_right, m_left, b_left, data['crosstrack'], data['heading'], data['downtrack']])

        if (verbose) and (i % 100 == 0):
            print("\t%d of %d" % (i, len(labels)))

    return X, y


def downsample_image(img, right_labels, left_labels):
    rightx1, righty1, rightx2, righty2 = right_labels
    leftx1, lefty1, leftx2, lefty2 = left_labels

    img_cropped = img[504:, :, :]
    img_grayscale = np.array(Image.fromarray(
        img_cropped).convert('L')) / 255.0

    factor_h = 1 / strideh
    factor_w = 1 / stridew

    img_ds = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img_ds[i, j] = np.mean(np.sort(
                img_grayscale[strideh*i:strideh*(i+1), stridew*j:stridew*(j+1)].reshape(-1))[-numPix:])

    right_line = np.cross(np.array([factor_w * rightx1, factor_h * (righty1 - 504), 1]),
                          np.array([factor_w * rightx2, factor_h * (righty2 - 504), 1]))

    left_line = np.cross(np.array([factor_w * leftx1, factor_h * (lefty1 - 504), 1]),
                         np.array([factor_w * leftx2, factor_h * (lefty2 - 504), 1]))

    return img_ds, right_line, left_line


def to_angle_offset(line):
    a, b, c = line
    phi = np.arctan(b / a)
    rho = c / np.sqrt(a**2 + b**2)
    return rho, phi


if __name__ == "__main__":
    X, y = get_data(dataset_dir)
    with h5py.File(os.path.join(dataset_dir, outfile), 'w') as f:
        f.create_dataset('X_train', data=X)
        f.create_dataset('y_train', data=y)
