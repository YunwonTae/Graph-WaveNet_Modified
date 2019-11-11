import pandas as pd
import numpy as np
import csv, math
from datetime import datetime
import os
import pickle
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default='/home/ytae/nas_datasets/traffic/incident/',help='data path')
parser.add_argument('--data_link',type=str,default='/home/ytae/traffic/main_roads.csv',help='data link path')
parser.add_argument('--save',type=str,default='./experiment/',help='save data')
args = parser.parse_args()

def main():
    final_df = pd.DataFrame({})
    city = 'kuangju'
    period = ['1905']
    for i, p in enumerate(period):
        print(city + " : " + p)
        df = pd.read_csv(args.data_path+'utic_'+city+'_dataframe_'+p+'.csv',engine='python')
        print("Read done")

        df.date = df.date.astype(str)

        idxs = df.loc[(df['date'].str[8:12] < '0600') | (df['date'].str[8:12] > '2100')].index
        df.drop(idxs, inplace=True)

        new_date = list(map(lambda n: datetime.strptime(str(n), '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'), df['date']))
        df['new_date'] = new_date
        df = df.set_index('new_date')
        df = df.drop(['date'], axis=1)

        df2 = pd.read_csv(args.data_link, header=None)
        routelist = list(map(str,df2.iloc[:,0]))
        df = df[routelist]

        df.index = pd.DatetimeIndex(df.index)
        df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="5min"), fill_value="NaN")

        df.index = df.index.astype(str)
        idxs = df.loc[(df.index.str[11:16] < '06:00') | (df.index.str[11:16] > '21:00')].index
        df.drop(idxs, inplace=True)

        count_nan = df.isna().sum()

        new_routelist = []
        for i,num_nan in enumerate(count_nan):
            if num_nan < 200:
                print(num_nan)
                link = count_nan.index[i]
                new_routelist.append(link)

        df = df[new_routelist]

        final_df = pd.concat([final_df, df], axis=0, sort=False)

    final_df = final_df.fillna(method = 'pad')
    final_df = final_df.fillna(method = 'bfill')

    pickle.dump(new_routelist, open(args.save+"routelist.pkl", "wb" ))
    final_df.to_hdf(args.save+'utic_'+city+'_dataframe.h5', key='DCRNN')
    print("finish")

if __name__ == "__main__":
    main()
