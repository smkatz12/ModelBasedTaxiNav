# Script for generating runway images using X-Plane 11 with edge labels
import time
import numpy as np
import random

import mss
import cv2
from PIL import Image

from xpc3 import *
from xpc3_helper_sm import *

import tqdm

OUTDIR = '/scratch/smkatz/class/CS231A/E16Data/'

# 0=Clear, 1=Cirrus, 2=Scattered, 3=Broken, 4=Overcast (higher numbers are cloudier/darker)
CLOUD_TYPE = 2
TIME_OF_DAY = 9.0  # 9am

# Number of images to grab
NUM_POINTS_CTE = 40
NUM_POINTS_DTP = 250

# From ground truth
left_edge_1 = [35929.9796875, -190.48041382, 46803.98203125]
left_edge_2 = [35753.2046875, -185.55480957, 46404.71953125]
right_edge_1 = [35950.821875, -190.52763977, 46794.65]
right_edge_2 = [35773.7703125, - 185.58735657, 46395.5515625]

def get_image_coord(pt, mv, proj, sh, sw):
    pt_h = np.append(np.array(pt), 1.0)
    pt_eye = mv @ pt_h
    pt_ndc = proj @ pt_eye
    pt_ndc = pt_ndc[:3] / pt_ndc[3]
    xp = sw * (pt_ndc[0] * 0.5 + 0.5)
    yp = sh - sh * (pt_ndc[1] * 0.5 + 0.5)
    return xp, yp

def get_edge_endpoints(client, img):
    sh, sw, _ = img.shape
    mv = np.reshape(client.getDREF("sim/graphics/view/world_matrix"), (4, 4)).T
    proj = np.reshape(client.getDREF(
        "sim/graphics/view/projection_matrix_3d"), (4, 4)).T
    left_x1, left_y1 = get_image_coord(left_edge_1, mv, proj, sh, sw)
    left_x2, left_y2 = get_image_coord(left_edge_2, mv, proj, sh, sw)
    right_x1, right_y1 = get_image_coord(right_edge_1, mv, proj, sh, sw)
    right_x2, right_y2 = get_image_coord(right_edge_2, mv, proj, sh, sw)
    return left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2

def generate_dataset(client, outdir, ncte, ndtp):
    # Reset simulator
    reset(client)

    client.sendDREF("sim/time/zulu_time_sec", TIME_OF_DAY*3600+8*3600)
    client.sendDREF("sim/weather/cloud_type[0]", CLOUD_TYPE)

    # Give a few seconds to get terminal out of the way
    time.sleep(2)

    # Screenshot parameters
    screen_shot = mss.mss()

    # Create label file
    csv_file = outdir + 'labels.csv'
    with open(csv_file, 'w') as fd:
        fd.write("filename,crosstrack,heading,downtrack,leftx1,lefty1,leftx2,lefty2,rightx1,righty1,rightx2,righty2\n")

    # Generate random downtrack positions for data
    dtps = [random.uniform(250, 350) for _ in range(ndtp)]
    dtps.sort()

    idx = 0
    for dtp in tqdm.tqdm(dtps):
        ctes = [random.uniform(-7, 7) for _ in range(ncte)]
        for cte in ctes:
            he = random.uniform(-30, 30)

            setHomeState(client, cte, dtp, he)
            time.sleep(0.25)
            img = cv2.cvtColor(np.array(screen_shot.grab(
                screen_shot.monitors[2])), cv2.COLOR_BGRA2BGR)
            cv2.imwrite('%s%d.png' % (outdir, idx), img)
            leftx1, lefty1, leftx2, lefty2, rightx1, righty1, rightx2, righty2 = get_edge_endpoints(client, img)

            with open(csv_file, 'a') as fd:
                fd.write("%d.png,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" %
                         (idx, cte, he, dtp, leftx1, lefty1, leftx2, lefty2, rightx1, righty1, rightx2, righty2))
            idx += 1
            

def main():
    with xpc3.XPlaneConnect() as client:
    	generate_dataset(client, OUTDIR, NUM_POINTS_CTE, NUM_POINTS_DTP)


if __name__ == "__main__":
	main()
