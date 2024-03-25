# /*
#  * @Author: wyuhui
#  * @Date: 2024-03-24 14:46:28
#  * @Last Modified by:   wyuhui
#  * @Last Modified time: 2024-03-24 14:46:28
#  */

import os
import pandas as pd
import re
from tqdm import trange
from tqdm import tqdm


class Params():
    def __init__(self, project_base_dir, project_name, area_name, result_save_dir):
        self.project_name = project_name
        self.project_dir = os.path.join(project_base_dir, project_name)
        self.area_name = area_name
        self.result_save_dir = result_save_dir
        self.src_data_dir = os.path.join(self.project_dir, area_name)
        return

    def update_positions(self, positions):
        self.positions = positions


""" 从文件名中读取源节点的编号 """


def read_idx_from_filename(params, filename, read_type):
    # 从文件名中读取源节点编号
    pattern = re.compile(
        fr'{re.escape(params.project_name)}.{re.escape(read_type)}.t(\d+)_(\d+).r001.p2m'
    )
    match = pattern.match(filename)
    if not match:
        return False
    else:
        return int(match.group(1)) - 1  # 节点编号, (从0开始)


""" 读取节点位置信息 """


def read_position(params):
    positions = []  # 节点位置
    for filename in tqdm(os.listdir(params.src_data_dir), desc='loading positions...'):
        if not filename.endswith('.p2m'):
            continue
        point_idx = read_idx_from_filename(
            params, filename, 'power')  # (从 power 文件中读取)
        if point_idx is False:
            continue

        file_path = os.path.join(params.src_data_dir, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            position_line = lines[point_idx + 3]  # +3 是取决于文件存储的方式
            [_, x, y, z, distance, power, _] = position_line.split(' ')
            positions.append([x, y])

    pd.DataFrame(positions, columns=['x', 'y']).to_csv(
        os.path.join(params.result_save_dir, 'positions.csv'))


""" 读取特定 type 类型的数据 """


def read_data_by_type(params: Params, data: pd.DataFrame, read_type: str):
    # 仿真输出文件的格式是, 以pointA为发, 其他所有节点为收的所有数据记录在一个以A节点编号命名的文件中
    is_data_empty = data.shape[0] == 0  # 初次读取时, 需记录源/目标节点序号和位置
    current_data = []  # 当前读取类型的data

    for filename in tqdm(os.listdir(params.src_data_dir), desc=f'loading {read_type}'):
        if not filename.endswith('.p2m'):
            continue
        pointA_idx = read_idx_from_filename(params, filename, read_type)
        if pointA_idx is False:
            continue

        if is_data_empty:
            src_idx = pointA_idx
            position = params.positions.iloc[src_idx]
            [tx_x, tx_y] = [position['x'], position['y']]
        file_path = os.path.join(params.src_data_dir, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()[3:]   # 根据文件存储格式,前三行为注释文字
            for line in lines:
                [dst_idx, rx_x, rx_y, rx_z, distance,
                    aim_data] = line.strip().split(' ')[:6]
                dst_idx = int(dst_idx) - 1  # 使其从0开始,保持全局一致
                if is_data_empty:
                    current_data.append(
                        [src_idx, dst_idx, tx_x, tx_y, rx_x, rx_y, distance, aim_data])
                    columns = f'src,dst,tx_x,tx_y,rx_x,rx_y,distance,{read_type}'.split(
                        ',')
                else:
                    current_data.append([aim_data])
                    columns = [read_type]

    return pd.concat((data, pd.DataFrame(current_data, columns=columns)), axis=1)


def main():
    project_base_dir = 'D:\wireless_projects'
    project_name = 'project_170'
    area_name = 'area2'  # 仿真软件中设定的区域名称
    result_save_dir = os.path.join('./data', project_name)
    os.makedirs(result_save_dir, exist_ok=True)
    params = Params(project_base_dir, project_name, area_name, result_save_dir)

    read_position(params)
    positions = pd.read_csv(os.path.join(
        params.result_save_dir, 'positions.csv'), index_col=0)
    params.update_positions(positions)

    data = pd.DataFrame([])  # 待写入的 data

    read_types = ['power', 'mtoa', 'pl', 'txloss']
    for type in read_types:
        data = read_data_by_type(params, data, type)
    data.to_csv(os.path.join(params.result_save_dir, 'data.csv'))


if __name__ == '__main__':
    main()
