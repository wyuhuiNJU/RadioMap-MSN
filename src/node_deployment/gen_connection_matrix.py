# /*
#  * @Author: wyuhui 
#  * @Date: 2024-03-25 20:36:23 
#  * @Last Modified by:   wyuhui 
#  * @Last Modified time: 2024-03-25 20:36:23 
#  */

from N_node_topo import Params, gen_connection_matrix
import pandas as pd
import os

def main():
    params = Params()
    point_num = 209
    project_dir = './data/project_small_209'

    params.set_project_params(point_num, project_dir)
    data = pd.read_csv(os.path.join(params.project_dir, 'data.csv'), index_col=0)

    connection_matrix = pd.DataFrame(gen_connection_matrix(params, data))
    columns = range(params.N)
    connection_matrix.to_csv(os.path.join(project_dir, 'connection_matrix.csv'), columns=columns, index=columns)
    print('done')

if __name__ == "__main__":
    main()
