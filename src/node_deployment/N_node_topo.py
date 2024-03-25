# /*
#  * @Author: wyuhui 
#  * @Date: 2024-03-25 16:41:55 
#  * @Last Modified by:   wyuhui 
#  * @Last Modified time: 2024-03-25 16:41:55 
#  */

import pandas as pd
import cv2
import os
import numpy as np

class Params():
    def __init__(self) -> None:
        pass

    def set_project_params(self, point_num, project_dir):
        self.N = point_num
        self.project_dir = project_dir
        pass

    def set_topo(self, nodeNum, draw_pic=False):
        self.nodeNum = nodeNum # 拓扑节点数量
        self.draw_pic = draw_pic # 生成拓扑时是否绘图

    def set_restriction(self, min_distance, min_dBm):
        self.min_distance = min_distance
        self.min_dBm = min_dBm
        pass

    def set_topo_param(self, topo_dir_name):
        self.topo_dir = os.path.join(self.project_dir, topo_dir_name)

def get_idx_in_data(src, dst, params: Params):
    return src * params.N + dst

# 生成全部点位的互联关系矩阵
def gen_connection_matrix(params: Params, data):
    N = params.N
    matrix = np.zeros((N, N))
    for src in range(N):
        for dst in range(src+1, N):
            src2dst_idx = get_idx_in_data(src, dst)
            dst2src_idx = get_idx_in_data(dst, src)
            distance = data.iloc[src2dst_idx]['distance']
            src2dst_power = data.iloc[src2dst_idx]['power']
            dst2src_power = data.iloc[dst2src_idx]['power']

            if params.min_distance is False: # 未设置约束条件, 获取原始的互联关系矩阵
                matrix[src][dst] = src2dst_power
                matrix[dst][src] = dst2src_power
            else:
                # 不满足约束的, 跳过
                if distance < params.min_distance:
                    continue
                if src2dst_power < params.min_dBm or dst2src_power < params.min_dBm:
                    continue
                # 满足约束
                matrix[src][dst] = src2dst_power
                matrix[dst][src] = dst2src_power
    return matrix

def gen_n_node_topo(connection_matrix, node_num: int, data, params: Params):
    matrix_save_dir = os.path.join(params.topo_dir, '2', 'matrix')
    if os.path.exists(matrix_save_dir):
        print(f'{node_num} node topo already exists')
        return
    os.makedirs(matrix_save_dir)
    if node_num < 2:
        raise Exception(f'拓扑节点数量不能小于2')
    counter = 0 # 拓扑数量计数
    if node_num == 2: # 双节点连接的矩阵, 从原始数据中生成
        for src in range(params.N):
            for dst in range(src+1, params.N): # 向后遍历,避免重复
                columns = [src, dst] # 矩阵的列名
                if not bool(connection_matrix(src)[dst]): # 未连接
                    continue
                matrix = np.zeros((2, 2))
                matrix[0][1] = connection_matrix[src][dst]
                matrix[1][0] = connection_matrix[dst][src]
                matrix_df = pd.DataFrame(matrix, columns=columns, index=columns)
                matrix_df.to_csv(os.path.join(matrix_save_dir, f'{counter}.csv'))
                counter += 1
        print(f'2 nodes connection topo: {counter}')
    else: # 节点数 n > 2, 则由 n-1 拓扑逐级生成
        matrix_read_dir = os.path.join(params.topo_dir, f'{node_num-1}', 'matrix')
        for filename in os.listdir(matrix_read_dir): #  N-1 拓扑的邻接矩阵
            pre_matrix = pd.read_csv(os.path.join(matrix_read_dir, filename), index_col=0)
            pre_nodes = list(map(int, pre_matrix.columns.tolist())) # N-1 拓扑的节点
            for next_node in range(max(pre_nodes)+1, params.N): # 向后遍历, 避免重复
                new_columns = pre_nodes + [next_node]
                new_matrix = np.zeros((node_num, node_num))
                new_matrix[:node_num-1, :node_num-1] += pre_matrix.to_numpy() # 新矩阵继承自原矩阵
                new_matrix = pd.DataFrame(new_matrix, columns=new_columns, index=new_columns)

                exceed_distance_limit = False # 超过距离限制
                connected_counter = 0 # 与原拓扑产生的连接数量
                for pre_node in pre_nodes:
                    distance = data.iloc[get_idx_in_data(pre_node, next_node)]['distance']
                    





    # matrix_read_dir = os.path.join(params.topo_dir, )
    

def main():
    params = Params()

    project_dir = './data/project_170'
    params.set_project_params(point_num=170, project_dir=project_dir)

    min_distance = 10 # 单位:米
    min_dBm = -50
    params.set_restriction(min_distance, min_dBm)

    topo_dir_name = 'topo-1'
    params.set_topo_param(topo_dir_name)

    data = pd.read_csv(os.path.join(params.project_dir))
    connection_matrix = gen_connection_matrix(params, data)
    
    

    pass