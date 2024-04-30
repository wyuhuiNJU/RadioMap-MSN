# /*
#  * @Author: wyuhui 
#  * @Date: 2024-03-29 14:48:08 
#  * @Last Modified by:   wyuhui 
#  * @Last Modified time: 2024-03-29 14:48:08 
#  */


import cv2
import numpy as np
import os
import pandas as pd

project_dir = './data/project_209'

# N = 170
# area_width = 80
# area_height = 50
# scaler = 20


N = 209
area_width = 50
area_height = 30
scaler = 20


positions = pd.read_csv(os.path.join(project_dir, 'positions.csv'), index_col=0)
xs = positions.iloc[:, 0].tolist()
ys = positions.iloc[:, 1].tolist()


img_width = area_width * scaler
img_height = area_height * scaler
img = 255 - np.zeros((img_height, img_width, 3), dtype=np.uint8)

point_color = (0, 0, 0)
radius = 10 # 圆点直径
thickness = 2 # 圆点边的厚度

for i in range(N):
    x, y = int(xs[i] * scaler), int(ys[i] * scaler)
    cv2.circle(img, (x, y), 1, point_color, thickness)
    cv2.putText(img, str(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4, color=point_color, bottomLeftOrigin=True)

# 170
# walls = [  # 墙壁两端坐标(包括围墙)
#     [(0, 0), (0, 80)],
#     [(0, 80), (80, 50)],
#     [(80, 50), (0, 50)],
#     [(0, 50), (0, 0)],
#     [(19, 50), (19, 22)],
#     [(40, 34), (40, 0)],
#     [(61, 22), (61, 50)],
#     # [(36, 30), (36, 21)],
# ]

# 209
walls = [  # 墙壁两端坐标(包括围墙)
    [(0, 0), (0, 50)],
    [(0, 50), (50, 30)],
    [(50, 30), (0, 30)],
    [(0, 30), (0, 0)],
    [(11, 30), (11, 7)],
    [(26, 0), (26, 24)],
    [(50, 11), (31, 11)],
    [(36, 30), (36, 21)],
]


for wall in walls:
    [point1, point2] = wall
    point1 = [int(point1[0] * scaler), int(point1[1] * scaler)]
    point2 = [int(point2[0] * scaler), int(point2[1] * scaler)]
    cv2.line(img, point1, point2, point_color, thickness)

img = cv2.flip(img, 0) # 上下翻转, 使其与建模视角相同
cv2.imwrite(os.path.join(project_dir, 'layout.jpg'), img)