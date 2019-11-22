from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle

import pdb


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)

    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[float(sensor_id)] = i

    # Fills cells in the matrix with distances.
    for idx,row in distance_df.iterrows():
        print(idx)
        if row[0] not in sensor_id_to_ind:
            continue

        for idx2,row2 in distance_df.iterrows():
            if row2[0] not in sensor_id_to_ind:
                continue
            if row[2] == row2[1]:
                if row2[2]==row[1]:
                    continue
                #TODO ##각 대학마다 포멧이 다를수있음##
                # dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row2[0]]] = (row[3]+row2[3]) / 2
                dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row2[0]]] = (row[19]+row2[19]) / 2


    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='../1.create_routelist/final_routelist.pkl',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='/home/ytae/traffic/main_roads.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='../data/adj_mx.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    # link select
    # sensor_ids = pd.read_csv(args.sensor_ids_filename).iloc[:, 1:].columns.tolist()
    # sensor_ids=[int(x) for x in sensor_ids]
    with open(args.sensor_ids_filename,'rb') as f:
        sensor_ids = pickle.load(f)
        sensor_ids = list(map(int, sensor_ids))

    # distance select
    # distance_df = pd.read_csv(args.distances_filename, engine='python',index_col=0)
    # drop_col = distance_df.iloc[:,3:-1].columns.tolist()
    # distance_df=distance_df.drop(drop_col,axis=1)
    # pdb.set_trace()

    #----------------------------------------------------------------------------------
    #TODO 전체 주요도로 링크 파일 => ##각 대학마다 포멧이 다를수있음##
    distance_df = pd.read_csv(args.distances_filename, engine='python',header=None)
    drop_col = distance_df.iloc[:,3:-1].columns.tolist()
    drop_col.append(30)
    drop_col.remove(19)
    distance_df=distance_df.drop(drop_col,axis=1)
    #----------------------------------------------------------------------------------


    # link slice
    # sensor_ids = sensor_ids[:200]

    # # distance slice
    # idx=distance_df.set_index(distance_df.columns[0])
    # idx=idx.loc[sensor_ids]
    # distance_df=idx.reset_index()
    # distance_df=distance_df.fillna(0)

    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
    print(adj_mx)


    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
