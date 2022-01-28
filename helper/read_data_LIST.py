import sys
import os
import pickle as pkl
from copy import deepcopy

[sys.path.append(i) for i in ['.', '..']]
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import argparse
from sklearn.metrics.pairwise import cosine_similarity


def load_pickle(path, name):
    with open(path + name, 'rb') as f:
        return pkl.load(f, encoding='latin1')


def similarity(matrix):
    sim_u = cosine_similarity(matrix, dense_output=False)
    sim_i = cosine_similarity(matrix.T, dense_output=False)

    return sim_u, sim_i


class Data(object):

    def __init__(self, dataset, pkl_path, pad=None, val_ratio=None, test_ratio=0.2, seed=0):

        self.dataset = dataset
        self.data_dir = pkl_path

        self.user_item_ratings = load_pickle(self.data_dir, 'csr_rating_mat.pkl')

        if val_ratio:
            val_ratio = val_ratio / (1 - test_ratio)

        self.train_matrix, self.test_matrix, self.val_matrix, \
        self.train_set, self.test_set, self.val_set = self.create_train_test_split(test_ratio=test_ratio,
                                                                                   val_ratio=val_ratio, seed=0)
        self.num_user, self.num_item = self.train_matrix.shape
        print('num_user: %d, num_item: %d' % (self.num_user, self.num_item))

        # self.u_adj_list, self.v_adj_list = self.create_adj_matrix(self.train_matrix, pad=pad)
        print(np.sum([len(x) for x in self.train_set]),
              np.sum([len(x) for x in self.val_set]),
              np.sum([len(x) for x in self.test_set]))

        avg_deg = np.sum([len(x) for x in self.train_set]) / self.num_user
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree U-I graph: ', avg_deg)

        if not self.val_set:
            full_set = [x + y for x, y in zip(self.train_set, self.test_set)]
        else:
            full_set = [x + y + z for x, y, z in zip(self.train_set, self.val_set, self.test_set)]

        full_matrix = self.generate_rating_matrix(full_set, self.num_user, self.num_item)

    def generate_inverse_mapping(self, data_list):
        ds_matrix_mapping = dict()
        for inner_id, true_id in enumerate(data_list):
            ds_matrix_mapping[true_id] = inner_id
        return ds_matrix_mapping

    def split_data_randomly(self, user_records, val_ratio, test_ratio, seed=0):
        train_set = []
        test_set = []
        val_set = []
        for user_id, item_list in enumerate(user_records):
            tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)

            if val_ratio:
                tmp_train_sample, tmp_val_sample = train_test_split(tmp_train_sample, test_size=val_ratio,
                                                                    random_state=seed)

            if val_ratio:
                train_sample = []
                for place in item_list:
                    if place not in tmp_test_sample and place not in tmp_val_sample:
                        train_sample.append(place)

                val_sample = []
                for place in item_list:
                    if place not in tmp_test_sample and place not in tmp_train_sample:
                        val_sample.append(place)

                test_sample = []
                for place in tmp_test_sample:
                    if place not in tmp_train_sample and place not in tmp_val_sample:
                        test_sample.append(place)

                train_set.append(train_sample)
                val_set.append(val_sample)
                test_set.append(test_sample)

            else:
                train_sample = []
                for place in item_list:
                    if place not in tmp_test_sample:
                        train_sample.append(place)

                test_sample = []
                for place in tmp_test_sample:
                    if place not in tmp_train_sample:
                        test_sample.append(place)

                train_set.append(train_sample)
                test_set.append(test_sample)

        return train_set, test_set, val_set

    def create_train_test_split(self, val_ratio=None, test_ratio=0.2, seed=0):
        if self.dataset == 'ml_100k':
            data_set = []
            for item_ids in self.user_item_ratings:
                data_set.append(item_ids.indices.tolist())

            train_set, test_set, val_set = self.split_data_randomly(data_set, val_ratio=val_ratio,
                                                                    test_ratio=test_ratio, seed=seed)
            train_matrix = self.generate_rating_matrix(train_set, self.user_item_ratings.shape[0],
                                                       self.user_item_ratings.shape[1])
            test_matrix = self.generate_rating_matrix(test_set, self.user_item_ratings.shape[0],
                                                      self.user_item_ratings.shape[1])
            val_matrix = self.generate_rating_matrix(val_set, self.user_item_ratings.shape[0],
                                                     self.user_item_ratings.shape[1])

        elif self.dataset in ['Amazon_CDs', 'Amazon_Books', 'Amazon_Movie', 'ml_10m', 'Amazon_Elec', 'Amazon_Beauty']:
            user_records = load_pickle(self.data_dir, 'csr_rating_mat.pkl')
            user_mapping = load_pickle(self.data_dir, self.dataset.split('_')[1] + '_user_mapping.pkl')
            item_mapping = load_pickle(self.data_dir, self.dataset.split('_')[1] + '_item_mapping.pkl')
            inner_data_records, user_inverse_mapping, \
            item_inverse_mapping = self.convert_to_inner_index(user_records, user_mapping, item_mapping)
            train_set, test_set, val_set = self.split_data_randomly(inner_data_records, val_ratio=val_ratio,
                                                                    test_ratio=test_ratio, seed=seed)

            train_matrix = self.generate_rating_matrix(train_set, len(user_mapping), len(item_mapping))
            test_matrix = self.generate_rating_matrix(test_set, len(user_mapping), len(item_mapping))
            val_matrix = self.generate_rating_matrix(val_set, len(user_mapping), len(item_mapping))

        elif self.dataset in ['good_comics', 'good_comics_sparse', 'yelp2018', 'gowalla']:
            user_records = load_pickle(self.data_dir, 'csr_rating_mat.pkl')

            train_set, test_set, val_set = self.split_data_randomly(self.convert_to_lists(user_records),
                                                                    val_ratio=val_ratio,
                                                                    test_ratio=test_ratio, seed=seed)
            num_users, num_items = user_records.shape
            train_matrix = self.generate_rating_matrix(train_set, num_users, num_items)
            test_matrix = self.generate_rating_matrix(test_set, num_users, num_items)
            val_matrix = self.generate_rating_matrix(val_set, num_users, num_items)

        else:
            raise NotImplementedError('NOT DATASET %s ' % self.dataset)

        return train_matrix, test_matrix, val_matrix, train_set, test_set, val_set

    def generate_rating_matrix(self, train_set, num_users, num_items, user_shift=0, item_shift=0):
        row = []
        col = []
        data = []
        for user_id, article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id + user_shift)
                col.append(article + item_shift)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)

        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
        return rating_matrix

    def convert_to_inner_index(self, user_records, user_mapping, item_mapping):
        inner_user_records = []
        user_inverse_mapping = self.generate_inverse_mapping(user_mapping)
        item_inverse_mapping = self.generate_inverse_mapping(item_mapping)

        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            item_list = list(user_records[real_user_id])
            for index, real_item_id in enumerate(item_list):
                item_list[index] = item_inverse_mapping[real_item_id]
            inner_user_records.append(item_list)

        return inner_user_records, user_inverse_mapping, item_inverse_mapping

    def convert_to_lists(self, user_records):
        inner_user_records = []
        user_records = user_records.A

        for user_id in range(user_records.shape[0]):
            item_list = np.where(user_records[user_id] != 0)[0].tolist()
            inner_user_records.append(item_list)

        return inner_user_records
