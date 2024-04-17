# Note 1: because of using the kmeans to reorder samples before training,
# the clustering performance is sensitive to the reoder to some extent.
from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn import Linear
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import scipy.io
from idecutils import cluster_acc, rebuild
import idecutils
from queue import Queue
from models import AE_2views as AE, FusionNet
import os
import time
import random
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
manual_seed = 0
os.environ['PYTHONHASHSEED'] = str(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def wmse_loss(input, target, reduction='mean'):
    ret = (target - input) ** 2
    ret = torch.mean(ret)
    return ret


def completion_mse(input, target, index):
    input_expend = torch.zeros_like(target)
    for i in range(target.shape[0]):
        input_expend[i] = input[index[i]]
    ret = (target - input_expend) ** 2
    ret = torch.mean(ret)
    return ret


class CRTC_2VIEW(nn.Module):

    def __init__(self,
                 n_input,
                 pretrain_path0='',
                 pretrain_path1=''):
        super(CRTC_2VIEW, self).__init__()
        self.pretrain_path0 = pretrain_path0
        self.pretrain_path1 = pretrain_path1

        self.fn0 = FusionNet(input_size=n_input[0])
        self.fn1 = FusionNet(input_size=n_input[1])

        self.fn0.cuda()
        self.fn1.cuda()

    def pretrain(self, path=''):
        if args.pretrain_CRTC_flag == 0:
            pretrain_crtc(self.fn0, self.fn1)
            print('pretrained ae finished')
            args.pretrain_CRTC_flag = 1
        else:
            self.fn0.load_state_dict(torch.load(self.pretrain_path0))
            self.fn1.load_state_dict(torch.load(self.pretrain_path1))
            print('load pretrained CRTC model from', self.pretrain_path0, self.pretrain_path1)

    def forward(self, x0, x1):
        output0 = self.fn0(x0)
        output1 = self.fn1(x1)
        return output0, output1


def pretrain_crtc(model0, model1):
    for m in model0.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    for m in model1.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    optimizer0 = Adam(model0.parameters(), lr=args.lrg, weight_decay=5e-4)
    optimizer1 = Adam(model1.parameters(), lr=args.lrg, weight_decay=5e-4)

    print('Pre-train view-0 FN')
    for epoch in range(args.graph_pretrain_epoch):
        model0.train()
        optimizer0.zero_grad()
        output0 = model0(Dropped0_resT)
        loss_train = completion_mse(output0, Knn0_dictT, Knn0_indexT)
        print('epoch:', epoch, 'loss:', loss_train.item())
        loss_train.backward()
        optimizer0.step()

    print('Pre-train view-1 FN')
    for epoch in range(args.graph_pretrain_epoch):
        model1.train()
        optimizer1.zero_grad()
        output1 = model1(Dropped1_resT)
        loss_train = completion_mse(output1, Knn1_dictT, Knn1_indexT)
        print('epoch:', epoch, 'loss:', loss_train.item())
        loss_train.backward()
        optimizer1.step()

    torch.save(model0.state_dict(), args.pretrain_path0)
    torch.save(model1.state_dict(), args.pretrain_path1)
    print("model saved to {}/{}.".format(args.pretrain_path0, args.pretrain_path1))


class MFC(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 n_clusters,
                 v=1,
                 pretrain_path=''):
        super(MFC, self).__init__()
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z)

        self.v = v
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if args.pretrain_MFC_flag == 0:
            pretrain_mfc(self.ae)
            print('pretrained ae finished')
            args.pretrain_MFC_flag = 1
        else:
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            print('load pretrained ae model from', self.pretrain_path)

    def forward(self, x0, x1):
        _, _, z, vz0, vz1 = self.ae(x0, x1)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, q, vz0, vz1


def pretrain_mfc(model):
    print(model)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    optimizer = SGD(model.parameters(), lr=args.lrae, momentum=0.95)
    #    model.train()
    index_array = np.arange(X0.shape[0])
    np.random.shuffle(index_array)
    loss_q = Queue(maxsize=50)
    for epoch in range(args.ae_pretrain_epoch):
        total_loss = 0.
        for batch_idx in range(np.int_(np.ceil(X0.shape[0] / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, X0.shape[0])]
            x0 = X0[idx].to(device)
            x1 = X1[idx].to(device)

            optimizer.zero_grad()
            x0_bar, x1_bar, hidden, vz0, vz1 = model(x0, x1)
            loss = wmse_loss(x0_bar, x0) + wmse_loss(x1_bar, x1)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_q.put(total_loss)
        if loss_q.full():
            loss_q.get()
        mean_loss = np.mean(list(loss_q.queue))
        if np.abs(mean_loss - total_loss) <= 0.001 and epoch >= 1000:
            print('Training stopped: epoch=%d, loss=%.4f, loss=%.4f' % (
                epoch, total_loss / (batch_idx + 1), mean_loss / (batch_idx + 1)))
            break
        print("ae_epoch {} loss={:.8f} mean_loss={:.8f}".format(epoch,
                                                                total_loss / (batch_idx + 1),
                                                                mean_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_MFC_path)
    print("model saved to {}.".format(args.pretrain_MFC_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--knn', default=10, type=int, help='number of nodes for subgraph embedding')
    parser.add_argument('--drop_index', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lrg', default=0.001, type=float)
    parser.add_argument('--lrae', default=0.01, type=float)
    parser.add_argument('--lrkl', default=0.0001, type=float)
    parser.add_argument('--cluster_max_epoch', default=100, type=int)
    parser.add_argument('--ae_pretrain_epoch', default=300, type=int)
    parser.add_argument('--graph_pretrain_epoch', default=100, type=int)
    parser.add_argument('--tol', default=1e-7, type=float)
    parser.add_argument('--t', default=50, type=int)
    # data and dir
    parser.add_argument('--pretrain_CRTC_flag', type=int, default=1)
    parser.add_argument('--pretrain_MFC_flag', type=int, default=1)
    parser.add_argument('--percentDel', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='BDGP_2view')
    parser.add_argument('--basis_pretrain_path', type=str, default='./save_weight/BDGP_2view/')
    parser.add_argument('--basis_save_dir', type=str, default='./data/BDGP_2view')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_CRTC_path = args.basis_pretrain_path + 'crtc_' + 'k_' + str(args.knn) + '_percentDel_0.' + str(
        args.percentDel) + '_glr_' + str(args.lrg) + '_ae_pretrain_epoch_' + str(args.graph_pretrain_epoch)

    args.pretrain_MFC_path = args.basis_pretrain_path + 'mfc_' + 'k_' + str(args.knn) + '_percentDel_0.' + str(
        args.percentDel) + '_aelr_' + str(args.lrae) + '_ae_pretrain_epoch_' + str(
        args.ae_pretrain_epoch) + '.pkl'

    ####################################################################
    # Load data, label, incomplete_index_matrix, and distance_matrix
    ####################################################################

    data = scipy.io.loadmat(args.basis_save_dir + '/' + args.dataset + '.mat')

    label = data['Ya']
    label = label.reshape(-1)
    label = np.array(label, 'float64')
    args.n_clusters = len(np.unique(label))
    y = label

    X = [data['x1'], data['x2']]

    dropMatrix = np.load(args.basis_save_dir + '/' + args.dataset + '_percentDel_0.' + str(args.percentDel) + '.npy',
                 allow_pickle=True).item()
    WE = dropMatrix[args.drop_index]

    distanceM = np.load(args.basis_save_dir + '/' + args.dataset + '_disMat.npy', allow_pickle=True).item()

    ####################################################################
    # Construct dropped distanceMatrix
    ####################################################################

    disMat = {}
    for i in range(len(distanceM)):
        del_index = np.array(np.where(WE[:, i] == 0)).squeeze()
        # idx_del_dict[i] = np.delete(idx_dict[i], del_index, 0)
        final_shape = np.delete(distanceM[i], del_index, 0).shape[0]
        disMat[i] = np.zeros((distanceM[i].shape[0], final_shape))
        for ii in range(disMat[i].shape[0]):
            if ii not in del_index:
                disMat[i][ii] = np.delete(distanceM[i][ii], np.where(distanceM[i][ii] == del_index[:, None])[1])
    del label, data, distanceM, dropMatrix, del_index

    ####################################################################
    # Construct dropped distanceMatrix
    ####################################################################
    # view-specific data 79, 1750 features
    X0 = np.array(X[0], 'float64')
    X1 = np.array(X[1], 'float64')
    args.n_input = [X0.shape[1], X1.shape[1]]
    del X
    # For each view,
    iv = 0
    # obtain drop and exist array
    WEiv = np.copy(WE[:, iv])
    WEiv0 = WEiv.copy()
    ind_0_complete = np.where(WEiv == 1)
    ind_0_complete = (np.array(ind_0_complete)).reshape(-1)
    ind_0_dropped = np.where(WEiv == 0)
    ind_0_dropped = (np.array(ind_0_dropped)).reshape(-1)
    # obtain the adj of dropping sample
    temp_dis_dict0 = {}
    for i in ind_0_dropped:
        exist_view = (np.array(np.where(WE[i] == 1))).reshape(-1)
        j = 0
        for ii in np.array(exist_view):
            if j == 0:
                temp_dis_dict0[i] = []
                temp_dis_dict0[i].append(disMat[ii][i][1:args.knn + 1].astype(int))
                j += 1
            else:
                temp_dis_dict0[i].append(disMat[ii][i][1:args.knn + 1].astype(int))
        if len(temp_dis_dict0[i]) == 2:
            temp_dis_dict0[i] = np.concatenate((temp_dis_dict0[i][0], temp_dis_dict0[i][1]))
        elif len(temp_dis_dict0[i]) == 3:
            temp_dis_dict0[i] = np.concatenate((temp_dis_dict0[i][0], temp_dis_dict0[i][1], temp_dis_dict0[i][2]))
        elif len(temp_dis_dict0[i]) == 4:
            temp_dis_dict0[i] = np.concatenate((temp_dis_dict0[i][0], temp_dis_dict0[i][1], temp_dis_dict0[i][2],
                                                temp_dis_dict0[i][3]))
        temp_dis_dict0[i] = np.setdiff1d(temp_dis_dict0[i], ind_0_dropped, True)
    # normalize
    X0[ind_0_complete, :] = StandardScaler().fit_transform(X0[ind_0_complete, :])
    X0[ind_0_dropped, :] = 0
    X0_re, Knn0_dict, Knn0_index, Dropped0_res = rebuild(X0, ind_0_dropped, temp_dis_dict0)

    iv = 1
    # obtain drop and exist array
    WEiv = np.copy(WE[:, iv])
    WEiv1 = WEiv.copy()
    ind_1_complete = np.where(WEiv == 1)
    ind_1_complete = (np.array(ind_1_complete)).reshape(-1)
    ind_1_dropped = np.where(WEiv == 0)
    ind_1_dropped = (np.array(ind_1_dropped)).reshape(-1)
    # obtain the adj of dropping sample
    temp_dis_dict1 = {}
    for i in ind_1_dropped:
        exist_view = (np.array(np.where(WE[i] == 1))).reshape(-1)
        j = 0
        for ii in np.array(exist_view):
            if j == 0:
                temp_dis_dict1[i] = []
                temp_dis_dict1[i].append(disMat[ii][i][1:args.knn + 1].astype(int))
                j += 1
            else:
                temp_dis_dict1[i].append(disMat[ii][i][1:args.knn + 1].astype(int))
        if len(temp_dis_dict1[i]) == 2:
            temp_dis_dict1[i] = np.concatenate((temp_dis_dict1[i][0], temp_dis_dict1[i][1]))
        elif len(temp_dis_dict1[i]) == 3:
            temp_dis_dict1[i] = np.concatenate((temp_dis_dict1[i][0], temp_dis_dict1[i][1], temp_dis_dict1[i][2]))
        elif len(temp_dis_dict1[i]) == 4:
            temp_dis_dict1[i] = np.concatenate((temp_dis_dict1[i][0], temp_dis_dict1[i][1], temp_dis_dict1[i][2],
                                                temp_dis_dict1[i][3]))
        temp_dis_dict1[i] = np.setdiff1d(temp_dis_dict1[i], ind_1_dropped, True)
    # normalize
    X1[ind_1_complete, :] = StandardScaler().fit_transform(X1[ind_1_complete, :])
    X1[ind_1_dropped, :] = 0
    # patching missing data
    X1_re, Knn1_dict, Knn1_index, Dropped1_res = rebuild(X1, ind_1_dropped, temp_dis_dict1)

    del iv, WEiv

    #######################################################
    # Pre-train CTRC
    #######################################################
    # pre-process data
    Knn0_dictT = torch.Tensor(np.array(Knn0_dict)).to(device)
    Knn1_dictT = torch.Tensor(np.array(Knn1_dict)).to(device)

    Knn0_indexT = np.array(Knn0_index)
    Knn1_indexT = np.array(Knn1_index)

    Dropped0_res = np.nan_to_num(Dropped0_res)
    Dropped1_res = np.nan_to_num(Dropped1_res)

    Dropped0_resT = torch.Tensor(Dropped0_res).to(device)
    Dropped1_resT = torch.Tensor(Dropped1_res).to(device)

    Dropped0_resT, Dropped1_resT = Variable(Dropped0_resT), Variable(Dropped1_resT)

    args.pretrain_path0 = args.pretrain_CRTC_path + '_2_0' + '.pkl'
    args.pretrain_path1 = args.pretrain_CRTC_path + '_2_1' + '.pkl'

    modelCTRC = CRTC_2VIEW(n_input=args.n_input, pretrain_path0=args.pretrain_path0, pretrain_path1=args.pretrain_path1)
    modelCTRC.pretrain()
    x0o, x01 = modelCTRC(Dropped0_resT, Dropped1_resT)
    X0_res = X0.copy()
    X0_res[ind_0_dropped] = x0o.data.cpu().numpy()
    X1_res = X1.copy()
    X1_res[ind_1_dropped] = x01.data.cpu().numpy()

    #######################################################
    # Pre-train MFC
    #######################################################

    X0 = torch.Tensor(X0_res).to(device)
    X1 = torch.Tensor(X1_res).to(device)

    model = MFC(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.n_clusters,
        n_clusters=args.n_clusters,
        pretrain_path=args.pretrain_MFC_path).to(device)
    model.pretrain()

    #######################################################
    # obtain the k-means clustering assignments based on the rebuild data
    #######################################################

    X0_res = np.nan_to_num(X0_res)
    X1_res = np.nan_to_num(X1_res)

    X_total = np.concatenate((X0_res, X1_res), axis=1)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)

    y_pred = kmeans.fit_predict(X0_res)
    del X_total, kmeans
    # re-order according to the pre-assignment, requiring the same class together
    X0_train = np.zeros(X0_res.shape)
    X1_train = np.zeros(X1_res.shape)

    label_train = np.zeros(y.shape)
    basis_index = 0
    for li in range(args.n_clusters):
        index_li = np.where(y_pred == li)
        index_li = (np.array(index_li)).reshape(-1)
        X0_train[np.arange(len(index_li)) + basis_index, :] = np.copy(X0_res[index_li])
        X1_train[np.arange(len(index_li)) + basis_index, :] = np.copy(X1_res[index_li])
        label_train[np.arange(len(index_li)) + basis_index] = np.copy(y[index_li])
        basis_index = basis_index + len(index_li)

    del X0, X1, WE, y
    X0 = np.copy(X0_train)
    X1 = np.copy(X1_train)
    y = label_train
    del X0_train, X1_train, X0_res, X1_res, label_train, basis_index, index_li

    #######################################################
    # TRAINING
    #######################################################
    X0 = torch.Tensor(X0).to(device)
    X1 = torch.Tensor(X1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lrkl)
    hidden, q, _, _ = model(X0, X1)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
    hidden = np.nan_to_num(hidden.data.cpu().numpy())
    y_pred = kmeans.fit_predict(hidden)
    del hidden
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    y_pred_last = y_pred

    best_acc2 = 0
    best_nmi2 = 0
    best_ari2 = 0
    best_f12 = 0
    best_epoch = 0
    total_loss = 0

    for epoch in range(int(args.cluster_max_epoch)):

        if epoch % 1 == 0:
            _, tmp_q, _, _ = model(X0, X1)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            y_pred = tmp_q.cpu().numpy().argmax(1)
            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            if acc > best_acc2:
                best_acc2 = np.copy(acc)
                best_nmi2 = np.copy(nmi)
                best_ari2 = np.copy(ari)
                best_epoch = epoch
            print('best_Iter {}'.format(best_epoch), ':best_Acc2 {:.4f}'.format(best_acc2), 'Iter {}'.format(epoch),
                  ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari), 'total_loss {:.8f}'.format(total_loss))
            total_loss = 0
            # check stop criterion
            delta_y = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if epoch > 80 and delta_y < args.tol:
                print('Training stopped: epoch=%d, delta_label=%.8f, tol=%.8f' % (epoch, delta_y, args.tol))
                break

        y_pred = torch.tensor(y_pred)
        index_array = np.arange(X0.shape[0])
        for batch_idx in range(np.int_(np.ceil(X0.shape[0] / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, X0.shape[0])]
            x0 = X0[idx].to(device)
            x1 = X1[idx].to(device)
            optimizer.zero_grad()
            hidden, q, vz0, vz1 = model(x0, x1)

            if np.isnan(hidden.data.cpu().numpy()).any():
                break
            kl_loss = F.kl_div(q.log(), p[idx], reduction='batchmean')
            fusion_loss = kl_loss
            total_loss += fusion_loss
            fusion_loss.backward()
            optimizer.step()

