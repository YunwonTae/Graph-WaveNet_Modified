import argparse
import torch.optim as optim
import util
from model import *
import numpy as np
import pandas as pd
import datetime
pd.options.display.float_format = '{:.8f}'.format
import yaml
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',default=True,help='whether add adaptive adj')
parser.add_argument('--randomadj',default=True,help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=977,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
# Graph embedding experiment
parser.add_argument('--graph_embedding', type=bool, default=False, help='True: graph_embedding, False: Original')
parser.add_argument('--graph_emb_update', type=bool, default=False, help='update graph_emb parameters')
# Transfer learning experiment
parser.add_argument('--load_model', type=bool, default=False, help='load model')
# Incident data experiment
parser.add_argument('--incident', type=bool, default=True, help='Train with incident features')
# Weather data experiment
parser.add_argument('--weather', type=bool, default=True, help='Train with weather features')

args = parser.parse_args()

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return [(d * self.std) + self.mean for d in data]

class trainer2():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, incident=False, weather=False, graph_emb_update=False, graph_emb=None):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, incident=incident, weather=weather, graph_emb_update=graph_emb_update, graph_emb=graph_emb)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

def generate_incident(config,df_input):
    data = np.expand_dims(df_input.values, axis=-1)
    return data

def generate_weather(config,df_input):
    final_df_list = []
    #TODO: 기상 feature 별로 monthly aggregate
    for i, feature in enumerate(['speed','temperature','rain','wind','humidity','incident']):
        if feature == 'speed':
            pass
        elif feature == 'incident':
            pass
        else:
            print(i)
            final_df_list.append(df_input[i])
    data_list = []
    for df in final_df_list:
        data = np.expand_dims(df.values, axis=-1)
        data_list.append(data)
    data = np.concatenate(data_list, axis=-1)

    return data

def generate_train(config,df_input):

    # reshape the input vector
    data = np.expand_dims(df_input.values, axis=-1)
    data_list = [data]
    time_ind = (df_input.index.values - df_input.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [1, config['num_nodes'], 1]).transpose((2, 1, 0))
    data_list.append(time_in_day)
    data = np.concatenate(data_list, axis=-1)

    return data


def main():

    # select gpu
    device = torch.device(args.device)

    # load the config file
    with open('data/kuangju'+'.yaml') as f:
        config = yaml.load(f)

    # load the adjacency matrix
    adjdata = 'data/adj_mx.pkl'
    if os.path.isfile(adjdata):
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adjdata, 'doubletransition')
    else:
        print(adjdata + ": does not exist!")
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    adjinit = None
    graph_emb = None
    scaler = StandardScaler(mean=config['mean'], std=config['std'])
    weather_scaler = StandardScaler(mean=config['weather_mean'], std=config['weather_std'])

    args.num_nodes = config['num_nodes']

    engine = trainer2(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, args.incident, args.weather, args.graph_emb_update, graph_emb)

    # load the model
    engine.model.load_state_dict(torch.load(config['checkpoint']))
    engine.model.eval()
    print('The model has been loaded successfully')

    #TODO hearder none
    df_OL = pd.read_csv(config['data'])  # 데이터 형식 다시받아야함
    del df_OL['4'] # 강수형태 column 지우기
    del df_OL['Unnamed: 0'] # 데이터 형식 다시받아야함

    features = ['speed','temperature','rain','wind','humidity','incident']
    df_OL.columns = ['date','link','speed','temperature','rain','wind','humidity','incident']

    data_frame_list = []
    for feature in features:
        df_temp = df_OL.pivot_table(index='date', values=feature, columns='link')
        df_temp = df_temp.fillna(method='ffill')
        df_temp = df_temp.fillna(method='bfill')
        data_frame_list.append(df_temp)
    assert len(data_frame_list) == len(features) #dataframe수와 freature수 같아야함

    df_input_list = []
    # 실시간 데이터에 링크가 없는 부분들은 평균으로 채워넣기
    for i,feature in enumerate(features):
        if feature == 'incident': #돌발은 0혹은 1이기 때문에 새로운 링크는 0으로 예외 처리
            df_input = pd.DataFrame(columns=sensor_ids)
            df_input = pd.concat([df_input, data_frame_list[i]], join_axes = [df_input.columns])
            df_input = df_input.fillna(0)
            df_input = df_input[sensor_ids]
            ix = df_input.index.tolist()
            date = []
            for i in ix:
                date.append(datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
            df_input.index = date
        else:
            df_input = pd.DataFrame(columns=sensor_ids)
            df_input = pd.concat([df_input, data_frame_list[i]], join_axes = [df_input.columns])
            df_input = df_input.fillna(np.nanmean(df_input.values,axis=1)[0])
            df_input = df_input[sensor_ids]
            ix = df_input.index.tolist()
            date = []
            for i in ix:
                date.append(datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
            df_input.index = date

        df_input_list.append(df_input)

    datax = generate_train(config,df_input_list[0])
    datax = np.expand_dims(datax, axis=0)

    incidentx = generate_incident(config,df_input_list[-1])
    incidentx = np.expand_dims(incidentx, axis=0)

    weatherx = generate_weather(config,df_input_list)
    weatherx = np.expand_dims(weatherx, axis=0)

    ######################################################

    # maker scaler
    datax = scaler.transform(datax)
    weatherx = weather_scaler.transform(weatherx)

    datax = torch.Tensor(datax).to(device)
    datax = datax.transpose(1, 3) # torch.Size([batch, feature, node, time_series])

    incidentx = torch.Tensor(incidentx).to(device)
    incidentx = incidentx.transpose(1, 3) # torch.Size([batch, feature, node, time_series])

    weatherx = torch.Tensor(weatherx).to(device)
    weatherx = weatherx.transpose(1, 3) # torch.Size([batch, feature, node, time_series])

    # Predict the speed after 5 minutes
    pred = engine.model(datax,incidentx,weatherx)
    pred = pred.squeeze().tolist()
    pred = [scaler.inverse_transform(p) for p in pred]
    df_pred = pd.DataFrame(pred, columns=sensor_ids)

    from datetime import  timedelta
    df_pred = df_pred.set_index(pd.Index([date[-1] + timedelta(minutes=5), date[-1] + timedelta(minutes=10),
                                                         date[-1] + timedelta(minutes=15), date[-1] + timedelta(minutes=20),
                                                         date[-1] + timedelta(minutes=25), date[-1] + timedelta(minutes=30),
                                                         date[-1] + timedelta(minutes=35), date[-1] + timedelta(minutes=40),
                                                         date[-1] + timedelta(minutes=45), date[-1] + timedelta(minutes=50),
                                                         date[-1] + timedelta(minutes=55), date[-1] + timedelta(minutes=60)]))

    df_pred.to_csv('data/prediction.csv', encoding='utf8')

    print('Prediction: ')
    print(df_pred)


if __name__ == "__main__":
    main()
