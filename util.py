import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import pdb
# python -m scripts.generate_embedding_data --output_dir=/home/ytae/nas_datasets/traffic/DCRNN/link_450 --graph_df_filename=/home/ytae/traffic/0807_utic_kuangju_sorted_by_distance.embeddings --traffic_df_filename=/home/ytae/traffic/data/utic_kuangju_dataframe_450_.h5

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class Incident_DataLoader(object):
    def __init__(self, xs, ys, incident, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param incident:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            incident_padding = np.repeat(incident[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            incident = np.concatenate([incident, incident_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.incident = incident

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, incident= self.xs[permutation], self.ys[permutation], self.incident[permutation]
        self.xs = xs
        self.ys = ys
        self.incident = incident

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                incident_i = self.incident[start_ind: end_ind, ...]
                incident_i = incident_i.astype(np.float)
                yield (x_i, y_i, incident_i)
                self.current_ind += 1

        return _wrapper()

class Weather_DataLoader(object):
    def __init__(self, xs, ys, weather, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param incident:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            weather_padding = np.repeat(weather[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            weather = np.concatenate([weather, weather_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.weather = weather

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, weather= self.xs[permutation], self.ys[permutation], self.weather[permutation]
        self.xs = xs
        self.ys = ys
        self.weather = weather

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                weather_i = self.weather[start_ind: end_ind, ...]
                weather_i = weather_i.astype(np.float)
                yield (x_i, y_i, weather_i)
                self.current_ind += 1

        return _wrapper()

class Incident_Weather_DataLoader(object):
    def __init__(self, xs, ys, incident, weather, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param incident:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            weather_padding = np.repeat(weather[-1:], num_padding, axis=0)
            incident_padding = np.repeat(incident[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            weather = np.concatenate([weather, weather_padding], axis=0)
            incident = np.concatenate([incident, incident_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.weather = weather
        self.incident = incident

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, weather, incident= self.xs[permutation], self.ys[permutation], self.weather[permutation], self.incident[permutation]
        self.xs = xs
        self.ys = ys
        self.weather = weather
        self.incident = incident

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                weather_i = self.weather[start_ind: end_ind, ...]
                weather_i = weather_i.astype(np.float)
                incident_i = self.incident[start_ind: end_ind, ...]
                incident_i = incident_i.astype(np.float)
                yield (x_i, y_i, incident_i, weather_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, graph_embedding=None, incident=None, weather=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir+'/velocity', category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    if incident == True and weather == True:
        for category in ['train', 'val', 'test']:
            incident_data = np.load(os.path.join(dataset_dir+'/incident', "incident_"+category + '.npz'))
            data['incident_' + category] = incident_data['x']
            weather_data = np.load(os.path.join(dataset_dir+'/weather', "weather_"+category + '.npz'))
            data['weather_' + category] = weather_data['x']
        weather_scaler = StandardScaler(mean=data['weather_train'][..., 0].mean(), std=data['weather_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            data['weather_' + category][..., 0] = weather_scaler.transform(data['weather_' + category][..., 0])
        data['train_loader'] = Incident_Weather_DataLoader(data['x_train'], data['y_train'], data['incident_train'], data['weather_train'], batch_size)
        data['val_loader'] = Incident_Weather_DataLoader(data['x_val'], data['y_val'], data['incident_val'], data['weather_val'], valid_batch_size)
        data['test_loader'] = Incident_Weather_DataLoader(data['x_test'], data['y_test'], data['incident_test'], data['weather_test'], test_batch_size)

    elif weather == True:
        for category in ['train', 'val', 'test']:
            weather_data = np.load(os.path.join(dataset_dir+'/weather', "weather_"+category + '.npz'))
            data['weather_' + category] = weather_data['x']
        weather_scaler = StandardScaler(mean=data['weather_train'][..., 0].mean(), std=data['weather_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            data['weather_' + category][..., 0] = weather_scaler.transform(data['weather_' + category][..., 0])
        data['train_loader'] = Weather_DataLoader(data['x_train'], data['y_train'], data['weather_train'], batch_size)
        data['val_loader'] = Weather_DataLoader(data['x_val'], data['y_val'], data['weather_val'], valid_batch_size)
        data['test_loader'] = Weather_DataLoader(data['x_test'], data['y_test'], data['weather_test'], test_batch_size)

    elif incident == True:
        for category in ['train', 'val', 'test']:
            incident_data = np.load(os.path.join(dataset_dir+'/incident', "incident_"+category + '.npz'))
            data['incident_' + category] = incident_data['x']
        data['train_loader'] = Incident_DataLoader(data['x_train'], data['y_train'], data['incident_train'], batch_size)
        data['val_loader'] = Incident_DataLoader(data['x_val'], data['y_val'], data['incident_val'], valid_batch_size)
        data['test_loader'] = Incident_DataLoader(data['x_test'], data['y_test'], data['incident_test'], test_batch_size)

    else:
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)

    data['scaler'] = scaler

    if graph_embedding:
        emb = np.load(os.path.join(dataset_dir, 'embedding.npz'))
        data['embedding'] = emb['embedding']

    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse
