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
parser.add_argument('--data',type=str,default='./data/',help='data path')
parser.add_argument('--data_link',type=str,default='../1.create_routelist/final_routelist.pkl',help='data link path')
args = parser.parse_args()

def main():
    city = 'kuangju'
    final_df = pd.DataFrame({})

    new_date1 = pd.date_range(start='1/1/2019', end='1/31/2019 23:55:00', freq="5min")
    new_date2 = pd.date_range(start='2/1/2019 00:05:00', end='2/28/2019 23:55:00', freq="5min")
    new_date3 = pd.date_range(start='3/1/2019 00:05:00', end='3/31/2019 23:55:00', freq="5min")
    new_date4 = pd.date_range(start='4/1/2019 00:05:00', end='4/30/2019 23:55:00', freq="5min")
    new_date5 = pd.date_range(start='5/1/2019 00:05:00', end='5/31/2019 23:55:00', freq="5min")
    new_date6 = pd.date_range(start='6/1/2019 00:05:00', end='6/30/2019 23:55:00', freq="5min")
    new_dates = [new_date1, new_date2, new_date3, new_date4, new_date5, new_date6]

    for month, new_date in enumerate(new_dates):
        print("Read:  " + args.data+'incident_' + str(month+1) +'.xls')
        # Reading incident file
        incident_xls = pd.read_excel(args.data+'incident_' + str(month+1) +'.xls')

        # Creating empty DataFrame which should be a same format as velocity DataFrame
        df = pd.DataFrame({})
        df["date"] = new_date
        df = df.set_index('date')

        # 1. 최종 선별된 링크를 사용하여 6개월치 datafrmae 만들기
        with open(args.data_link,'rb') as f:
            routelist = pickle.load(f)
        routelist = list(map(str,routelist))

        # Create one-hot encoding dataframe
        df1 = pd.DataFrame("0",index=df.index, columns=routelist)

        # Selects the main road links from the whole links
        incident_df = incident_xls[incident_xls['LINKID'].isin(routelist)]

        # flooring for REPORTDATE, ceiling for ENDDATE
        incident_df['STARTDATE'] = incident_df['STARTDATE'].dt.floor('5min')
        incident_df['ENDDATE'] = incident_df['ENDDATE'].dt.ceil('5min')

        for i, link_id in enumerate(incident_df['LINKID']):
            index = pd.date_range(start=incident_df['STARTDATE'].iloc[i], end=incident_df['ENDDATE'].iloc[i], freq="5min")
            df_idx = incident_df.iloc[i]['INCIDENTCODE']
            if df_idx != 0:
                df1.loc[index, str(int(link_id))] = 1

            else:
                print("Incident code should not be zero!")
                assert df_idx != 0

        print("Size of the " + str(month+1) +" month: ", df1.shape)

        df1.index = df1.index.astype(str)
        df1.to_csv(args.data+'incident_' + str(month+1) +'.csv')
        print("Finish:   " + str(month+1))

        final_df = pd.concat([final_df, df1], axis=0, sort=False)

    final_df.to_csv(args.data+'utic_'+city+'_incident_final.csv')
    print("finish")


if __name__ == "__main__":
    main()
