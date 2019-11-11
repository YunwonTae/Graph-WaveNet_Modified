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
parser.add_argument('--data',type=str,default='/home/ytae/nas_datasets/traffic/incident/incident_05.xls',help='data path')
parser.add_argument('--adj_mx',type=str,default='/home/ytae/nas_datasets/traffic/incident/experiment/adj_mx.pkl',help='data link path')
parser.add_argument('--save',type=str,default='./experiment',help='save data')
parser.add_argument('--start_date',type=str,default='5/1/2019', help='Please specify start date')
parser.add_argument('--end_date_time',type=str,default='5/31/2019 23:55:00', help='Please specify end date with time specific')
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
    incident_xls = pd.read_excel(args.data)

    # Creating empty DataFrame which should be a same format as velocity DataFrame
    df = pd.DataFrame({})
    new_date = pd.date_range(start=args.start_date, end=args.end_date_time, freq="5min")
    df["date"] = new_date
    df = df.set_index('date')

    # Reading routelist
    with open(args.adj_mx,'rb') as f:
        adj_mx = pickle.load(f)
        routelist, _, _ = adj_mx
    routelist = list(map(str,routelist))

    # Create one-hot encoding dataframe
    df1 = pd.DataFrame("1",index=df.index, columns=routelist)
    df2 = pd.DataFrame("0",index=df.index, columns=routelist)
    df3 = pd.DataFrame("0",index=df.index, columns=routelist)
    df4 = pd.DataFrame("0",index=df.index, columns=routelist)
    df5 = pd.DataFrame("0",index=df.index, columns=routelist)
    data_list = [df1,df2,df3,df4,df5]

    # Selects the main road links from the whole links
    incident_df = incident_xls[incident_xls['LINKID'].isin(routelist)]

    # flooring for REPORTDATE, ceiling for ENDDATE
    incident_df['REPORTDATE'] = incident_df['REPORTDATE'].dt.floor('5min')
    incident_df['ENDDATE'] = incident_df['ENDDATE'].dt.ceil('5min')

    for i, link_id in enumerate(incident_df['LINKID']):
        index = pd.date_range(start=incident_df['REPORTDATE'].iloc[i], end=incident_df['ENDDATE'].iloc[i], freq="5min")
        df_idx = incident_df.iloc[i]['INCIDENTCODE']
        if df_idx != 0:
            data_list[0].loc[index, str(int(link_id))] = 0
            data_list[df_idx].loc[index, str(int(link_id))] = 1
        else:
            print("Incident code should not be zero!")
            assert df_idx != 0

    for idx in range(len(data_list)):
        data_list[idx].index = data_list[idx].index.astype(str)
        idxs = data_list[idx].loc[(data_list[idx].index.str[11:16] < '06:00') | (data_list[idx].index.str[11:16] > '21:00')].index
        data_list[idx].drop(idxs, inplace=True)
        data_list[idx] = np.expand_dims(data_list[idx].values, axis=-1)

    data = np.concatenate(data_list, axis=-1)

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
