import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='/home/ytae/nas_datasets/traffic/region/1901',help='data path')
parser.add_argument('--adjdata',type=str,default='/home/ytae/nas_datasets/traffic/region/1901/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=341,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,default='/home/ytae/nas_datasets/traffic/region/1707_1807/experiment/341/341_exp5_best_8.02.pth',help='')
parser.add_argument('--plotheatmap',type=str,default='False',help='')
parser.add_argument('--graph_embedding', type=bool, default=False, help='graph_embedding')

parser.add_argument('--test_other', type=bool, default=False, help='test on different data')
parser.add_argument('--test_adj', type=str, default='/home/ytae/nas_datasets/traffic/region/1901/adj_mx.pkl', help='test on different data')

args = parser.parse_args()




def main():
    device = torch.device(args.device)

    _, route2idx, adj_mx = util.load_adj(args.adjdata,args.adjtype)

    if args.test_other:
        routelist = []
        routelist_, routelist2id, _ = util.load_adj(args.test_adj,args.adjtype)
        for route in routelist_:
            if route in route2idx:
                routelist.append(route2idx[route])
            else:
                print("Test route does not exist in original adj_mx")
                assert False

    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args.graph_embedding)

    # pdb.set_trace()

    args.graph_embedding = False

    if args.graph_embedding:
        graph_emb = dataloader['embedding']
        graph_emb = torch.Tensor(graph_emb.astype('float64')).to(device)

    if args.graph_embedding:
        model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16, graph_emb=graph_emb)
    else:
        model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16, graph_emb=None)

    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()


    print('model load successfully')
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    amae = []
    amape = []
    armse = []
    predictions = []
    y_truths = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        predictions.append(pred.cpu().detach().numpy())
        real = scaler.inverse_transform(realy[:,:,i])
        y_truths.append(real.cpu().detach().numpy())
        if args.test_other:
            metrics = util.metric(pred[:,routelist],real[:,routelist])
        else:
            metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))


    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb"+ '.pdf')

    # y12 = np.array(scaler.inverse_transform(realy[:,99,11]))
    # yhat12 = np.array(scaler.inverse_transform(yhat[:,99,11]))
    #
    # y3 = np.array(scaler.inverse_transform(realy[:,99,2]))
    # yhat3 = np.array(scaler.inverse_transform(yhat[:,99,2]))

    outputs = {
        'predictions': predictions,
        'groundtruth': y_truths
    }

    np.savez_compressed(args.data+"/wavenet_predictions", **outputs)
    # df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    # df2.to_csv('./wave.csv',index=False)


if __name__ == "__main__":
    main()