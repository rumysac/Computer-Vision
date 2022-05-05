# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:11:12 2021

@author: Rumeysa CELIK
"""

import numpy as np
import cv2

#Write a script file to demonstrate your Hough based line detector on live video.

def generateHoughLines(img, indices, rhos, thetas):
    for i in range(len(indices)):
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


def hough(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height ** 2 + width ** 2))
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(thetas)):
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1
    return H, rhos, thetas


def houghLocations(H, num_peaks):
    indices = []
    size = 10
    a = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(a)
        AInd = np.unravel_index(idx, a.shape)
        indices.append(AInd)

        idx_y, idx_x = AInd
        if (idx_x - (size / 2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (size / 2)
        if (idx_x + (size / 2) + 1) > H.shape[1]:
            max_x = H.shape[1]
        else:
            max_x = idx_x + (size / 2) + 1

        if (idx_y - (size / 2)) < 0:
            min_y = 0
        else:
            min_y = idx_y - (size / 2)
        if (idx_y + (size / 2) + 1) > H.shape[0]:
            max_y = H.shape[0]
        else:
            max_y = idx_y + (size / 2) + 1

        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                a[y, x] = 0
                if x == min_x or x == (max_x - 1):
                    H[y, x] = 255
                if y == min_y or y == (max_y - 1):
                    H[y, x] = 255
    return indices, H


if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    a = 0
    while True:
        a = a + 1
        check, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(frame, 100, 200)
        H, rhos, thetas = hough(edges)
        indices, H = houghLocations(H, 10)
        generateHoughLines(frame, indices, rhos, thetas)
        cv2.imshow("Captured Video", frame)
        cv2.imwrite('VideOutput.jpg', frame)
        key = cv2.waitKey(1)
        if key == ord('w'):
            break

    video.release()


