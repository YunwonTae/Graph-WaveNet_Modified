import pandas as pd
import numpy as np
import csv, math
import os
import pickle
import random
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,default='/home/ytae/nas_datasets/traffic/distributed_utic_dataframe/',help='data path')
parser.add_argument('--main_routelist',type=str,default='/home/ytae/traffic/main_roads.csv',help='data path')
parser.add_argument('--save',type=str,default='./final_routelist.pkl',help='data link path')
args = parser.parse_args()

def routelist_intersection(common_routelist, routelist):
    return set(routelist).intersection(set(common_routelist))

# 1. 전체 소통 데이터의 공통된 링크 찾기
# 2. 결측값이 200개 미만인 링크만 가져오기
# 3. 주요도로 링크와 겹치는 링크를 최종 링크로 선정
def main():
    city = 'kuangju'
    period = ['1901','1902','1903','1904','1905','1906']

    for index, p in enumerate(period):
        print(city + " : " + p)
        ft = pd.read_csv(args.data+'utic_'+city+'_dataframe_'+p+'.csv')
        print("read done")

        count_nan = ft.isna().sum()
        new_routelist = []
        for i,num_nan in enumerate(count_nan):
            if num_nan < 200: # 2. 결측값이 200개 미만인 링크만 가져오기
                link = count_nan.index[i]
                new_routelist.append(link)

        print("Length of current data routelist: ", len(new_routelist))
        if index == 0:
            common_routelist = new_routelist
            continue
        else:
            common_routelist = list(set(new_routelist).intersection(set(common_routelist)))
            print("common_routelist of routelist: ", len(common_routelist))

    #routelist intersection with main_routelist
    df2 = pd.read_csv(args.main_routelist, header=None) # 주요도로 읽어오기
    routelist = list(map(str,df2.iloc[:,0]))

    final_routelist = routelist_intersection(common_routelist, routelist) # 3. 주요도로 링크와 겹치는 링크를 최종 링크로 선정
    print("Number of final routelist: " + str(len(final_routelist)))

    pickle.dump(list(final_routelist), open(args.save, "wb" ))
    print("Finished")

if __name__ == "__main__":
    main()
