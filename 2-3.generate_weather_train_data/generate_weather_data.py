from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import pdb


def generate_graph_seq2seq_io_data(
        data, y_offsets, x_offsets, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    """
    num_samples, num_nodes, _ = data.shape
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x = []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        x.append(x_t)

    x = np.stack(x, axis=0)

    return x


def generate_train_val_test(args):
    final_df = pd.DataFrame({})
    final_df_list = []
    #TODO: 기상 feature 별로 monthly aggregate
    for feature in ['humidity', 'rain', 'temperature', 'wind']:
        final_df = pd.DataFrame({})
        for month in range(args.num_month):
            df = pd.read_csv(args.traffic_df_filename+ 'kuangju_main_'+str(month+1)+'_'+feature+'.csv')
            final_df = pd.concat([final_df, df], axis=0, sort=False)
        final_df_list.append(final_df)

    # pdb.set_trace()

    data_list = []
    for df in final_df_list:
        df = df.set_index(df.columns[0]) # 시간이 index로 설정 안되있을경우
        data = np.expand_dims(df.values, axis=-1)
        data_list.append(data)

    data = np.concatenate(data_list, axis=-1)

    # pdb.set_trace()

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )

    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    x = generate_graph_seq2seq_io_data(
        data,
        y_offsets=y_offsets,
        x_offsets=x_offsets,
    )

    print("x shape: ", x.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
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
            os.path.join(args.output_dir, "weather_"+"%s.npz" % cat),
            x=_x,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_month", type=int, default=6, help="Numer of months"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/home/nas_datasets/traffic/incident/data/weather", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="../4.weather/preprocessed/data/",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
