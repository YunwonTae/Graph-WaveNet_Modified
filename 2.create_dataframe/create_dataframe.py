import pandas as pd
import numpy as np
import csv, math
from datetime import datetime
import os
import pickle
import random
import argparse
import pdb

# import warnings
# warnings.filterwarnings('ignore',category=pandas.io.pytables.PerformanceWarning) # 알수없는 warning메세지 제거
# TODO numpy==1.15.4 버전 체크

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default='/home/nas_datasets/traffic/distributed_utic_dataframe/',help='data path')
parser.add_argument('--data_link',type=str,default='../1.create_routelist/final_routelist.pkl',help='data link path')
parser.add_argument('--save',type=str,default='./',help='save data')
args = parser.parse_args()

# 1. 최종 선별된 링크를 사용하여 6개월치 datafrmae 만들기
# 2. 결측치는 앞뒤 속력값으로 채우기
# 3. 중간중간 비워져 있는 시간 채우기
# 4. 전체 1~6월의 continuity 위해 0시 ~ 24시를 학습으로 사용
# 5. 각 월마다 저장
# 6. 전체 1~6월 aggregate된 데이터 저장

def main():
    final_df = pd.DataFrame({})
    city = 'kuangju'
    period = ['1901','1902','1903','1904','1905','1906']
    for i, p in enumerate(period):
        print(city + " : " + p)
        df = pd.read_csv(args.data_path+'utic_'+city+'_dataframe_'+p+'.csv')
        print("Read done")

        df.date = df.date.astype(str)
        # pdb.set_trace()

        #TODO 1902월 날짜 인덱스 패턴 고려하여 if statement 제거 혹은 사용
        if p == '1902':
            new_date = list(map(lambda n: datetime.strptime(str(n), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'), df['date']))
            df['new_date'] = new_date
            df = df.set_index('new_date')
            df = df.drop(['date'], axis=1)

        else:
            new_date = list(map(lambda n: datetime.strptime(str(n), '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'), df['date']))
            df['new_date'] = new_date
            df = df.set_index('new_date')
            df = df.drop(['date'], axis=1)

        # 1. 최종 선별된 링크를 사용하여 6개월치 datafrmae 만들기
        with open(args.data_link,'rb') as f:
            routelist = pickle.load(f)
        routelist = list(map(str,routelist))
        df = df[routelist]

        # 3. 중간중간 비워져 있는 시간 채우기
        df.index = pd.DatetimeIndex(df.index)
        df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="5min"), fill_value=np.nan)
        df.index = df.index.astype(str)

        # pdb.set_trace()

        print("Size of the " + p +" month: ", df.shape)

        # 5. 각 월마다 저장
        temp_df = df
        temp_df = temp_df.fillna(method = 'pad') # 2. 결측치는 앞뒤 속력값으로 채우기
        temp_df = temp_df.fillna(method = 'bfill') # 2. 결측치는 앞뒤 속력값으로 채우기
        assert temp_df.isna().sum().sum() == 0 #결측값이 존재하지 않아야함
        temp_df = temp_df.apply(pd.to_numeric)
        temp_df.to_hdf(args.save+'utic_'+city+'_'+p+'_dataframe.h5',key='a')

        # 전체 1~6월 aggregate
        final_df = pd.concat([final_df, df], axis=0, sort=False)

    # 2. 결측치는 앞뒤 속력값으로 채우기
    final_df = final_df.fillna(method = 'pad')
    final_df = final_df.fillna(method = 'bfill')
    assert final_df.isna().sum().sum() == 0 #결측값이 존재하지 않아야함

    # 6. 전체 1~6월 aggregate된 데이터 저장
    final_df.to_hdf(args.save+'utic_'+city+'_dataframe_final.h5',key='b')
    print("finish")

if __name__ == "__main__":
    main()
