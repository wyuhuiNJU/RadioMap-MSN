# /*
#  * @Author: wyuhui 
#  * @Date: 2024-03-29 14:52:24 
#  * @Last Modified by:   wyuhui 
#  * @Last Modified time: 2024-03-29 14:52:24 
#  */

# 把采样点根据物理位置分类, 赋予标签, 以便用分类的方法进行粗粒度定位
# 分类依据是用draw_points_layout.py绘制的 layout.jpg图片

# 自定义类别区域
""" 
以project_209/layout.jpg为例
0 -> 0-43
1 -> 44-109
2 -> 110-116,121-127,132-138,143-149
3 -> 154-160,165-171,176-182,187-193,198-204
4 -> 117-120,128-131,139-142,150-153,161-164,172-175,183-186,194-197,205-208
"""

import pandas as pd
import numpy as np
import os




def get_area_points_idx(left_top_point_idx, right_bottom_point_idx, width, height):
    column_diff = int((right_bottom_point_idx - left_top_point_idx - height + 1) / (width - 1))
    points_idx = []
    for column in range(width):
        column_begin_idx = left_top_point_idx + (column_diff * column)
        for row in range(height):
            points_idx.append(column_begin_idx + row)
    return points_idx

def set_label(points_idx, class_num, dataframe):
    for idx in points_idx:
        dataframe.iat[idx, -1] = class_num


def main():
    project_dir = './data/project_209'
    positions = pd.read_csv(os.path.join(project_dir, 'positions.csv'), index_col=0)
    positions['label'] = 0 # 添加 label 列
    class_1_points = get_area_points_idx(44, 109, 6, 11)
    set_label(class_1_points, 1, positions)
    class_2_points = get_area_points_idx(110, 149, 4, 7)
    set_label(class_2_points, 2, positions)
    class_3_points = get_area_points_idx(154, 204, 5, 7)
    set_label(class_3_points, 3, positions)
    class_4_points = get_area_points_idx(117, 208, 9, 4)
    set_label(class_4_points, 4, positions)

    positions.to_csv(os.path.join(project_dir, 'positions_labeled.csv'))

main()
    
