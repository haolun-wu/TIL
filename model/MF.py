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
        self.train_label = args.train_label

        self.user_embeddings = nn.Embedding(self.num_users, args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(self.num_items, args.dim).to(self.device)
        self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)

        self.agg_user_embeddings = nn.Embedding(self.num_users, args.dim).to(self.device)
        self.agg_item_embeddings = nn.Embedding(self.num_items, args.dim).to(self.device)
        self.agg_user_embeddings.weight.requires_grad = False
        self.agg_item_embeddings.weight.requires_grad = False

        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]

    def compute_cosine(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0, 1))
        return res

    def community_ui(self, user_id):
        batch_label = self.train_label[user_id]
        # if self.thres == 'N':
        pos_i_com = torch.matmul(batch_label, self.item_embeddings.weight)
        num = batch_label.sum(1).unsqueeze(1)
        pos_i_com = pos_i_com / num
        # elif self.thres == 'Y':
        #     batch_u_emb = self.agg_user_embeddings(user_id)
        #     batch_sim = self.compute_cosine(batch_u_emb, self.agg_item_embeddings.weight)
        #     pos_i_com = self.neighbor(batch_label, batch_sim, -0.8)

        return pos_i_com

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1, keepdim=True)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1, keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -nn.LogSigmoid()(tmp)
        # mf_loss = -torch.sum(maxi)

        return bpr_loss

    def forward(self, user, pos, neg):
        # thr: threshold

        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = \
            self.user_embeddings(user_id), self.item_embeddings(pos_id), self.item_embeddings(neg_id)

        pos_i_com = self.community_ui(user_id)

        self.agg_user_embeddings.weight[user_id] = user_emb.data
        self.agg_item_embeddings.weight[pos_id] = pos_emb.data
        self.agg_item_embeddings.weight[neg_id] = neg_emb.data

        return user_emb, pos_emb, neg_emb, pos_i_com

    def out_forward(self, user, pos, neg, user_embeddings, item_embeddings):
        # thr: threshold

        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = user_embeddings[user_id], item_embeddings[pos_id], item_embeddings[neg_id]

        # batch_ui_tensor = self.train_ui_tensor[user_id]
        # pos_i_com = torch.matmul(batch_ui_tensor, item_embeddings)
        # num = batch_ui_tensor.sum(1).unsqueeze(1)
        # pos_i_com = pos_i_com / num

        self.agg_user_embeddings.weight[user_id] = user_emb.data
        self.agg_item_embeddings.weight[pos_id] = pos_emb.data
        self.agg_item_embeddings.weight[neg_id] = neg_emb.data

        return user_emb, pos_emb, neg_emb

    def predict(self, user_id):
        user_emb = self.agg_user_embeddings(user_id)
        pred = user_emb.mm(self.agg_item_embeddings.weight.t())

        return pred

    def get_embeddings(self, ids, emb_name):
        if emb_name == 'user':
            return self.user_embeddings[ids]
        elif emb_name == 'item':
            return self.item_embeddings[ids]
        else:
            return None


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
