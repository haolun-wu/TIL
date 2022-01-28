import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import spatial

import time

epsilon = 1e-5


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(MatrixFactorization, self).__init__()

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.dim = args.dim
        # self.use_ng = args.use_ng
        self.train_label = args.train_label
        self.interact_idx = args.interact_idx
        # self.value_mask = args.value_mask
        # self.select_mask = args.select_mask
        self.cluster = args.cluster
        self.norm = args.norm
        self.scale1 = args.scale1
        self.scale2 = args.scale2

        self.user_embeddings = nn.Embedding(self.num_users, args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(self.num_items + 1, args.dim, padding_idx=self.num_items).to(self.device)
        self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)

        # self.cluster_center = nn.Embedding(self.cluster, args.dim).to(self.device)
        # self.cluster_center.weight.data = torch.nn.init.xavier_uniform_(self.cluster_center.weight.data)

        # self.item_embeddings.weight.data[self.num_items] = torch.FloatTensor([-float('inf')] * self.dim)

        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]

    def compute_cluster_center(self, cluster_ids):
        cluster_center_emb = None
        for i in range(self.cluster):
            index = torch.where(cluster_ids == i)[0]
            tmp_emb = torch.mean(self.item_embeddings(index), dim=0, keepdim=True)
            cluster_center_emb = tmp_emb if cluster_center_emb == None else torch.cat((cluster_center_emb, tmp_emb), 0)

        return cluster_center_emb

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1, keepdim=True)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1, keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        # mf_loss = -torch.sum(maxi)

        return bpr_loss

    @staticmethod
    def max_norm(param, max_val=1, eps=1e-8):
        norm = param.norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norm, 0, max_val)
        param = param * (desired / (eps + norm))

        return param

    def bpr_loss_plus(self, users, pos_items, neg_items, pos_center, neg_center):
        scale = self.scale1
        pos_scores = torch.sum(
            torch.mul(users, pos_items) + scale * torch.mul(users, pos_center) + scale * torch.mul(pos_items,
                                                                                                   pos_center),
            dim=1,
            keepdim=True)
        neg_scores = torch.sum(
            torch.mul(users, neg_items) + scale * torch.mul(users, neg_center) + scale * torch.mul(neg_items,
                                                                                                   neg_center),
            dim=1,
            keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        # mf_loss = -torch.sum(maxi)

        return bpr_loss

    def forward_person_plus(self, user, pos, neg, cluster_ids):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = self.user_embeddings(user_id), self.item_embeddings(pos_id), self.item_embeddings(
            neg_id)

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
        uni_center = torch.matmul(batch_label[:, :-1], self.item_embeddings.weight[:-1])
        num_rel = batch_label.sum(1).unsqueeze(1)
        uni_center = uni_center / num_rel

        # print("pos_person_center:", pos_person_center.grad_fn)
        # print("pos_i_com:", pos_i_com.grad_fn)

        return user_emb, pos_emb, neg_emb, pos_center, neg_center, uni_center

    # def forward_person_super(self, user, pos, neg, cluster_ids, P):
    #     user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
    #     pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
    #     neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)
    #
    #     user_emb, pos_emb, neg_emb = self.user_embeddings(user_id), self.item_embeddings(pos_id), self.item_embeddings(
    #         neg_id)
    #
    #     pos_item_cluster_id = cluster_ids[pos_id]
    #     neg_item_cluster_id = cluster_ids[neg_id]
    #
    #     # select the center for those pos_item
    #     pos_center = self.cluster_center.weight[pos_item_cluster_id]
    #     neg_center = self.cluster_center.weight[neg_item_cluster_id]
    #
    #     # unified center
    #     batch_label = self.train_label[user_id, :]
    #     uni_center = torch.matmul(batch_label[:, :-1], self.item_embeddings.weight.data[:-1])
    #     num_rel = batch_label.sum(1).unsqueeze(1)
    #     uni_center = uni_center / num_rel
    #
    #     item_ids = np.array(list(set(pos_id).union(set(neg_id)))).astype('int32')
    #     item_ids = torch.tensor(item_ids).type(torch.LongTensor).to(self.device)
    #
    #     Q = self.get_Q(self.item_embeddings.weight[item_ids], self.cluster_center.weight)
    #
    #     kl_loss = F.kl_div(Q.log(), P[item_ids], reduction='sum')
    #     # kl_loss = 0
    #
    #     # print("kl_loss:", kl_loss)
    #
    #     # print("pos_person_center:", pos_person_center.grad_fn)
    #     # print("pos_i_com:", pos_i_com.grad_fn)
    #
    #     return user_emb, pos_emb, neg_emb, pos_center, neg_center, uni_center, kl_loss

    def forward_unify(self, user, pos, neg):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = self.user_embeddings(user_id), self.item_embeddings(pos_id), self.item_embeddings(
            neg_id)

        return user_emb, pos_emb, neg_emb

    def out_forward(self, user, pos, neg, user_embeddings, item_embeddings, cluster_ids):
        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = user_embeddings[user_id], item_embeddings[pos_id], item_embeddings[neg_id]

        # compute updated cluster_center_emb
        cluster_center_emb = None
        for i in range(self.cluster):
            index = torch.where(cluster_ids == i)[0]
            tmp_emb = item_embeddings[index]
            cluster_center_emb = tmp_emb if cluster_center_emb == None else torch.cat((cluster_center_emb, tmp_emb), 0)

        pos_item_cluster_id = cluster_ids[pos_id]
        neg_item_cluster_id = cluster_ids[neg_id]
        pos_center = cluster_center_emb[pos_item_cluster_id]
        neg_center = cluster_center_emb[neg_item_cluster_id]

        # item_ids = np.array(list(set(pos_id).union(set(neg_id)))).astype('int32')
        # item_ids = torch.tensor(item_ids).type(torch.LongTensor).to(self.device)
        #
        # Q = self.get_Q(item_embeddings[item_ids], cluster_center_embeddings)
        #
        # kl_loss = F.kl_div(Q.log(), P[item_ids], reduction='sum')

        return user_emb, pos_emb, neg_emb, pos_center, neg_center

    def get_Q(self, z, center):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - center, 2), 2))
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def predict(self, user_id):
        user_emb = self.user_embeddings(user_id).data
        item_emb = self.item_embeddings.weight.data[:-1]

        pred = user_emb.mm(item_emb.t())

        return pred

    def predict_plus(self, user_id, cluster_ids):
        user_emb = self.user_embeddings(user_id).data
        item_emb = self.item_embeddings.weight.data[:-1]

        cluster_center_emb = self.compute_cluster_center(cluster_ids).data
        cluster_emb = cluster_center_emb[cluster_ids[:-1]]

        rating_ui = user_emb.mm(item_emb.t())
        rating_uc = user_emb.mm(cluster_emb.t())
        rating_ic = torch.sum(item_emb * cluster_emb, 1).expand_as(rating_ui)

        scale = self.scale2
        rating = rating_ui + scale * rating_uc + scale * rating_ic

        return rating


class Controller(nn.Module):
    def __init__(self, dim1, device):
        super(Controller, self).__init__()

        self.linear1 = nn.Linear(dim1, 64, bias=True).to(device)
        self.linear2 = nn.Linear(64, 1, bias=False).to(device)

    def forward(self, x):
        z1 = torch.relu(self.linear1(x))
        # res = F.sigmoid(self.linear2(z1))
        res = F.softplus(self.linear2(z1))

        return res
