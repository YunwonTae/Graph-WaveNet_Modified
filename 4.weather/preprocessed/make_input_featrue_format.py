import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,default='../resources/main/feature',help='data path')
parser.add_argument('--data_link',type=str,default='../../1.create_routelist/final_routelist.pkl',help='data link path')
args = parser.parse_args()

def main():

    path = args.path

    feature = os.listdir(path)
    new_date1 = pd.date_range(start='1/1/2019', end='1/31/2019 23:55:00', freq="5min")
    new_date2 = pd.date_range(start='2/1/2019 00:05:00', end='2/28/2019 23:55:00', freq="5min")
    new_date3 = pd.date_range(start='3/1/2019 00:05:00', end='3/31/2019 23:55:00', freq="5min")
    new_date4 = pd.date_range(start='4/1/2019 00:05:00', end='4/30/2019 23:55:00', freq="5min")
    new_date5 = pd.date_range(start='5/1/2019 00:05:00', end='5/31/2019 23:55:00', freq="5min")
    new_date6 = pd.date_range(start='6/1/2019 00:05:00', end='6/30/2019 23:55:00', freq="5min")
    new_dates = [new_date1, new_date2, new_date3, new_date4, new_date5, new_date6]
    month_index = [-1, 744, 1417, 2162, 2883, 3628, -1]
    # df['2'].index[df['2'].apply(np.isnan)] #month_index 찾기
    
    for month, new_date in enumerate(new_dates):
        df = pd.DataFrame({})
        df["date"] = new_date
        df = df.set_index('date')

        with open(args.data_link,'rb') as f:
            routelist = pickle.load(f)
        routelist = list(map(str,routelist))

        df2 = pd.DataFrame("0",index=df.index, columns=routelist)

        link_code = pd.read_csv('../resources/main/regional_code.csv', usecols=['name', 'code'])
        link_dict = dict(zip(link_code.name, link_code.code))

        print(feature)
        for f in feature:
            print(f)
            path = '../resources/main/feature'
            path = path + '/' + f
            region = os.listdir(path)

            for r in region:
                print(r)
                # pdb.set_trace()
                l_code = link_dict[r.split('.')[0]]

                # 지역 별 feature load
                fr = pd.read_csv(path + '/' +r) #, engine='python', dtype= {'format: day':str, 'hour':str, 'value location:51_129 Start : 20190501': float})
                new_fr = fr.iloc[month_index[month]+1:month_index[month+1]]
                new_fr.columns = ['0','1','2']
                keys = list(new_fr.index)
                mydict = dict(zip(keys, new_fr['2']))

                l_code = link_dict[r.split('.')[0]]
                region_routelist = []
                for route in routelist:
                    code = route[:3]
                    if int(code) == int(l_code):
                        region_routelist.append(route)

                # TODO 시간줄이기
                # pdb.set_trace()
                for index, key in enumerate(keys):
                    # print(index)
                    index_s = index*12
                    index_e = index_s+12
                    time_index = df.iloc[index_s:index_e].index
                    df2.at[time_index,region_routelist] = mydict[key]


            print(path + ":           Finish")
            temp = month+1
            df2.to_csv('./data/kuangju_main_' + str(temp) + '_' + f + '.csv')

if __name__ == "__main__":
    main()
