import pandas as pd
import numpy as np
import csv, math
from datetime import datetime
import os
import pickle
import random
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,default='../3.create_incident/data/utic_kuangju_incident_final.csv',help='data path')
parser.add_argument('--save',type=str,default='../data/incident',help='save data')
args = parser.parse_args()

def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, scaler=None
):
    """
    Generate samples from
    :param data:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    """

    num_samples, num_nodes, _ = data.shape
    x = []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        x.append(x_t)

    x = np.stack(x, axis=0)

    return x

def main():
    # Reading incident file
    print("Reading file")
    df = pd.read_csv(args.data)
    df = df.set_index(df.columns[0]) # 시간이 index로 설정 안되있을경우
    print("Reading done")

    data = np.expand_dims(df.values, axis=-1)

    # data = np.concatenate(data_list, axis=-1)

    x_offsets = np.sort( np.concatenate((np.arange(-11, 1, 1),)))

    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    x = generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )

    print("x shape: ", x.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train = x[:num_train]
    # val
    x_val = x[num_train: num_train + num_val]
    # test
    x_test = x[-num_test:]

    for cat in ["train", "val", "test"]:
        _x = locals()["x_" + cat]
        print(cat, "x: ", _x.shape)
        np.savez_compressed(
            os.path.join(args.save, "incident_"+"%s.npz" % cat),
            x=_x,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    main()
