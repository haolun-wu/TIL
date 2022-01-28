import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

epsilon = 1e-5


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, args):
        super(NGCF, self).__init__()
        self.n_users = n_user
        self.n_items = n_item
        self.n_fold = args.n_fold
        self.device = args.device
        self.emb_size = args.dim
        self.batch_size = args.batch_size
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = eval(args.node_dropout)
        self.mess_dropout = eval(args.mess_dropout)
        self.batch_size = args.batch_size
        self.R = args.R.astype('float32')
        self.model = args.model
        self.train_label = args.train_label

        plain_adj, norm_adj, mean_adj = self.get_adj_mat()
        self.norm_adj = norm_adj
        self.cluster = args.cluster

        self.layers = eval(args.layer_size)

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.user_embeddings = nn.Embedding(self.n_users, args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(self.n_items, args.dim).to(self.device)
        self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)

        self.weight_dict = self.init_weight()

        self.myparameters = [self.user_embeddings.weight,
                             self.item_embeddings.weight] + list(self.weight_dict.values())

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def get_adj_mat(self):
        adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        weight_dict = dict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            if self.model == 'n':
                weight_dict['W_gc_%d' % k] = initializer(torch.empty(layers[k], layers[k + 1])).to(self.device)
                weight_dict['b_gc_%d' % k] = initializer(torch.empty(1, layers[k + 1])).to(self.device)
                weight_dict['W_gc_%d' % k].requires_grad = True
                weight_dict['b_gc_%d' % k].requires_grad = True

                weight_dict['W_bi_%d' % k] = initializer(torch.empty(layers[k], layers[k + 1])).to(self.device)
                weight_dict['b_bi_%d' % k] = initializer(torch.empty(1, layers[k + 1])).to(self.device)
                weight_dict['W_bi_%d' % k].requires_grad = True
                weight_dict['b_bi_%d' % k].requires_grad = True
            elif self.model == 'm':
                weight_dict['W_gc_%d' % k] = initializer(torch.empty(layers[k], layers[k + 1])).to(self.device)
                weight_dict['b_gc_%d' % k] = initializer(torch.empty(1, layers[k + 1])).to(self.device)
                weight_dict['W_gc_%d' % k].requires_grad = True
                weight_dict['b_gc_%d' % k].requires_grad = True

                weight_dict['W_mlp_%d' % k] = initializer(torch.empty(layers[k], layers[k + 1])).to(self.device)
                weight_dict['b_mlp_%d' % k] = initializer(torch.empty(1, layers[k + 1])).to(self.device)
                weight_dict['W_mlp_%d' % k].requires_grad = True
                weight_dict['b_mlp_%d' % k].requires_grad = True
            elif self.model == 'g':
                weight_dict['W_gc_%d' % k] = initializer(torch.empty(layers[k], layers[k + 1])).to(self.device)
                weight_dict['b_gc_%d' % k] = initializer(torch.empty(1, layers[k + 1])).to(self.device)
                weight_dict['W_gc_%d' % k].requires_grad = True
                weight_dict['b_gc_%d' % k].requires_grad = True

        return weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _dropout_sparse(self, x, keep_prob, noise_shape):
        # keep_prob = 1 - drop_rate
        random_tensor = keep_prob
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * torch.div(1., keep_prob)

    def community_ui(self, user_id):
        batch_label = self.train_label[user_id]
        pos_i_com = torch.matmul(batch_label, self.item_embeddings.weight)
        num = batch_label.sum(1).unsqueeze(1)
        pos_i_com = pos_i_com / num

        return pos_i_com

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1, keepdim=True)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1, keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        # mf_loss = -torch.sum(maxi)

        return bpr_loss

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self, drop_flag=True):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), ego_embeddings))
            side_embeddings = torch.cat(temp_embed, 0)

            # transformed sum messages of neighbors.
            sum_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]
            )

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages ofn neighbors.
            bi_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]
            )

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings = all_embeddings[:self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users:, :]

        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self, drop_flag=True):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], 0)

        all_embeddings = [embeddings]

        for k in range(len(self.layers)):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), embeddings))

            embeddings = torch.cat(temp_embed, 0)
            embeddings = torch.matmul(embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]
            embeddings = nn.LeakyReLU(negative_slope=0.2)(embeddings)
            embeddings = nn.Dropout(1 - self.mess_dropout[k])(embeddings)

            all_embeddings += [embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users:, :]
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self, drop_flag=True):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], 0)

        all_embeddings = []

        for k in range(len(self.layers)):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), embeddings))

            embeddings = torch.cat(temp_embed, 0)
            # convolutional layer.
            embeddings = torch.matmul(embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]
            embeddings = nn.LeakyReLU(negative_slope=0.2)(embeddings)
            # dense layer.
            mlp_embeddings = torch.matmul(embeddings, self.weight_dict['W_mlp_%d' % k]) + self.weight_dict[
                'b_mlp_%d' % k]
            mlp_embeddings = nn.Dropout(self.mess_dropout[k])(mlp_embeddings)

            all_embeddings += [mlp_embeddings]
        all_embeddings = torch.cat(all_embeddings, 1)

        u_g_embeddings = all_embeddings[:self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users:, :]
        return u_g_embeddings, i_g_embeddings

    def _create_lightgcn_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], 0)

        all_embeddings = [embeddings]

        for k in range(len(self.layers)):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), embeddings))

            embeddings = torch.cat(temp_embed, 0)

            all_embeddings.append(embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        light_out = torch.mean(all_embeddings, dim=1)
        u_g_embeddings = light_out[:self.n_users, :]
        i_g_embeddings = light_out[self.n_users:, :]
        return u_g_embeddings, i_g_embeddings

    def computer(self):
        if self.model == 'n':
            u_g_embeddings, i_g_embeddings = self._create_ngcf_embed()
        elif self.model == 'g':
            u_g_embeddings, i_g_embeddings = self._create_gcn_embed()
        elif self.model == 'm':
            u_g_embeddings, i_g_embeddings = self._create_gcmc_embed()
        elif self.model == 'l':
            u_g_embeddings, i_g_embeddings = self._create_lightgcn_embed()

        return u_g_embeddings, i_g_embeddings

    def forward_unify(self, user_id, pos_id, neg_id):
        u_g_embeddings, i_g_embeddings = self.computer()

        user_emb = u_g_embeddings[user_id]
        pos_emb = i_g_embeddings[pos_id]
        neg_emb = i_g_embeddings[neg_id]

        return user_emb, pos_emb, neg_emb

    def compute_cluster_center(self, cluster_ids):
        cluster_center_emb = None
        for i in range(self.cluster):
            index = torch.where(cluster_ids == i)[0]
            tmp_emb = torch.mean(self.item_embeddings.weight[index], dim=0, keepdim=True)
            cluster_center_emb = tmp_emb if cluster_center_emb == None else torch.cat((cluster_center_emb, tmp_emb), 0)

        return cluster_center_emb

    def forward_person_plus(self, user, pos, neg, cluster_ids):

        u_g_embeddings, i_g_embeddings = self.computer()
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = u_g_embeddings[user_id], i_g_embeddings[pos_id], i_g_embeddings[
            neg_id]

        pos_item_cluster_id = cluster_ids[pos_id]
        neg_item_cluster_id = cluster_ids[neg_id]

        # select the center for those pos_item
        # pos_center = self.cluster_center.weight[pos_item_cluster_id]
        # neg_center = self.cluster_center.weight[neg_item_cluster_id]
        cluster_center_emb = self.compute_cluster_center(cluster_ids)
        pos_center = cluster_center_emb[pos_item_cluster_id]
        neg_center = cluster_center_emb[neg_item_cluster_id]

        # unified center
        batch_label = self.train_label[user_id, :]
        uni_center = torch.matmul(batch_label, self.item_embeddings.weight)
        num_rel = batch_label.sum(1).unsqueeze(1)
        uni_center = uni_center / num_rel

        user_emb_ego = self.user_embeddings.weight[user_id]
        pos_emb_ego = self.item_embeddings.weight[pos_id]
        neg_emb_ego = self.item_embeddings.weight[neg_id]

        return user_emb_ego, pos_emb_ego, neg_emb_ego, user_emb, pos_emb, neg_emb, pos_center, neg_center, uni_center

    def out_forward(self, user, pos, neg, user_embeddings, item_embeddings, drop_flag=True):

        # if self.model == 'n':
        #     if self.node_dropout_flag:
        #         A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        #     else:
        #         A_fold_hat = self._split_A_hat(self.norm_adj)
        #
        #     ego_embeddings = torch.cat([user_embeddings, item_embeddings], 0)
        #     all_embeddings = [ego_embeddings]
        #
        #     for k in range(len(self.layers)):
        #         temp_embed = []
        #         for f in range(self.n_fold):
        #             temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), ego_embeddings))
        #         side_embeddings = torch.cat(temp_embed, 0)
        #         sum_embeddings = nn.LeakyReLU(negative_slope=0.2)(
        #             torch.matmul(side_embeddings, weight_embeddings['W_gc_%d' % k]) + weight_embeddings['b_gc_%d' % k]
        #         )
        #
        #         bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
        #         bi_embeddings = nn.LeakyReLU(negative_slope=0.2)(
        #             torch.matmul(bi_embeddings, weight_embeddings['W_bi_%d' % k]) + weight_embeddings['b_bi_%d' % k]
        #         )
        #
        #         ego_embeddings = sum_embeddings + bi_embeddings
        #         ego_embeddings = nn.Dropout(1 - self.mess_dropout[k])(ego_embeddings)
        #         norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
        #         all_embeddings += [norm_embeddings]
        #     all_embeddings = torch.cat(all_embeddings, dim=1)
        # elif self.model == 'g':
        #     if self.node_dropout_flag:
        #         A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        #     else:
        #         A_fold_hat = self._split_A_hat(self.norm_adj)
        #     embeddings = torch.cat([user_embeddings, item_embeddings], 0)
        #
        #     all_embeddings = [embeddings]
        #
        #     for k in range(len(self.layers)):
        #         temp_embed = []
        #         for f in range(self.n_fold):
        #             temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), embeddings))
        #
        #         embeddings = torch.cat(temp_embed, 0)
        #         embeddings = torch.matmul(embeddings, weight_embeddings['W_gc_%d' % k]) + weight_embeddings[
        #             'b_gc_%d' % k]
        #         embeddings = nn.LeakyReLU(negative_slope=0.2)(embeddings)
        #         embeddings = nn.Dropout(1 - self.mess_dropout[k])(embeddings)
        #
        #         all_embeddings += [embeddings]
        #
        #     all_embeddings = torch.cat(all_embeddings, dim=1)
        #     u_g_embeddings = all_embeddings[:self.n_users, :]
        #     i_g_embeddings = all_embeddings[self.n_users:, :]
        # elif self.model == 'm':
        #     A_hat = self._dropout_sparse(self.sparse_norm_adj,
        #                                  1 - self.node_dropout[0],
        #                                  self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        #
        #     embeddings = torch.cat([user_embeddings, item_embeddings], 0)
        #
        #     all_embeddings = []
        #
        #     for k in range(len(self.layers)):
        #         embeddings = torch.sparse.mm(A_hat, embeddings)
        #         # convolutional layer.
        #         embeddings = torch.matmul(embeddings, weight_embeddings['W_gc_%d' % k]) + weight_embeddings[
        #             'b_gc_%d' % k]
        #         # dense layer.
        #         mlp_embeddings = torch.matmul(embeddings, weight_embeddings['W_mlp_%d' % k]) + weight_embeddings[
        #             'b_mlp_%d' % k]
        #         mlp_embeddings = nn.Dropout(1 - self.mess_dropout[k])(mlp_embeddings)
        #
        #         all_embeddings += [mlp_embeddings]
        #     all_embeddings = torch.cat(all_embeddings, dim=1)
        # elif self.model == 'l':
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = torch.cat([user_embeddings, item_embeddings], 0)

        all_embeddings = [embeddings]

        for k in range(len(self.layers)):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f].to(self.device), embeddings))

            embeddings = torch.cat(temp_embed, 0)

            all_embeddings.append(embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)

        light_out = torch.mean(all_embeddings, dim=1)
        u_g_embeddings = light_out[:self.n_users, :]
        i_g_embeddings = light_out[self.n_users:, :]

        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = u_g_embeddings[user_id], i_g_embeddings[pos_id], i_g_embeddings[neg_id]

        return user_emb, pos_emb, neg_emb

    def predict(self, user_id):
        u_g_embeddings, i_g_embeddings = self.computer()
        user_emb = u_g_embeddings[user_id]
        pred = user_emb.mm(i_g_embeddings.t())

        return pred


class Controller(nn.Module):
    def __init__(self, dim1, device):
        super(Controller, self).__init__()

        self.linear1 = nn.Linear(dim1, 64, bias=True).to(device)
        self.linear2 = nn.Linear(64, 1, bias=False).to(device)

        torch.nn.init.kaiming_normal_(self.linear1.weight)
        # torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        # torch.nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        z1 = torch.relu(self.linear1(x))
        res = F.sigmoid(self.linear2(z1))
        # res = F.softplus(self.linear2(z1))

        return res
