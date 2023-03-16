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

OUTDIR = '/scratch/smkatz/class/CS231A/E16Data_omstrajv2/'

# 0=Clear, 1=Cirrus, 2=Scattered, 3=Broken, 4=Overcast (higher numbers are cloudier/darker)
CLOUD_TYPE = 2
TIME_OF_DAY = 9.0  # 9am

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

def next_position(x, y, theta, v=5, dt=0.1):
     return x + v * np.sin(theta) * dt, y + v * np.cos(theta) * dt

def generate_dataset(client, outdir):
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
        fd.write(
            "filename,crosstrack,heading,downtrack,leftx1,lefty1,leftx2,lefty2,rightx1,righty1,rightx2,righty2\n")

    # Generate random downtrack positions for data
    dtp = 200.0 #0.0 #250.0
    cte = 0.0
    he = np.degrees(0.5 * np.cos(0.0))

    idx = 0
    for t in np.arange(0, 40, 0.1): #np.arange(0, 13, 0.1):
        # print(cte, dtp, he)
        setHomeState(client, cte, dtp, he)
        time.sleep(0.25)
        img = cv2.cvtColor(np.array(screen_shot.grab(
            screen_shot.monitors[2])), cv2.COLOR_BGRA2BGR)
        cv2.imwrite('%s%d.png' % (outdir, idx), img)
        leftx1, lefty1, leftx2, lefty2, rightx1, righty1, rightx2, righty2 = get_edge_endpoints(
            client, img)

        with open(csv_file, 'a') as fd:
            fd.write("%d.png,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" %
                        (idx, cte, he, dtp, leftx1, lefty1, leftx2, lefty2, rightx1, righty1, rightx2, righty2))
        he = np.degrees(0.5 * np.cos(0.5 * t))
        cte, dtp = next_position(cte, dtp, np.radians(he))
        idx += 1


def main():
    with xpc3.XPlaneConnect() as client:
    	generate_dataset(client, OUTDIR)


if __name__ == "__main__":
	main()
