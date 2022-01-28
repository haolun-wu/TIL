import sys
import os
import pickle as pkl
import pandas as pd
import torch
from copy import deepcopy

[sys.path.append(i) for i in ['.', '..']]
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import pickle


def generate_IT_matrix(df):
    df_IT = df[['item', 'tag']]
    df_IT = df_IT.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True)
    # print("df_genre:", df_genre)
    # df_IT = pd.concat([pd.Series(row['item'], row['genre'].split(',')) for _, row in df_genre.iterrows()]).reset_index()
    df_IT = pd.DataFrame([[i, t] for i, T in df_IT.values for t in T], columns=df_IT.columns)
    # c = df_IT.columns
    # df_IT[[c[0], c[1]]] = df_IT[[c[1], c[0]]]
    df_IT.columns = ['item', 'tag']

    return df_IT


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class Delicious(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'delicious_ui.csv')
        self.fpath_tag = os.path.join(data_dir, 'delicious_it.csv')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep=',',
                              engine='python',
                              names=['user', 'item'],
                              skiprows=1).drop_duplicates()
        df_tag = pd.read_csv(self.fpath_tag,
                             sep=',',
                             engine='python',
                             names=['item', 'tag'],
                             skiprows=1).drop_duplicates()

        return df_rate, df_tag


class Data(object):
    def __init__(self, data_dir, val_ratio=None, test_ratio=0.2, user_filter=10, item_filter=10, tag_filter=10, seed=0):
        df_UI, df_IT = Delicious(data_dir).load()
        # df = self.remove_infrequent_items(df, item_filter)
        # df = self.remove_infrequent_users(df, user_filter)
        # print("df_UI:", df_UI)
        # print("df_IT:", df_IT)
        # return

        df_IT = self.remove_infrequent_tags(df_IT, tag_filter)
        # df_UI = df[['user', 'item']]
        # df_IT = df[['item', 'tag']]

        df_IT = df_IT.groupby(['item'])['tag'].apply(list)

        # df_UI = df_UI.reset_index(drop=True)
        df_IT = df_IT.reset_index()

        # print("merging...")
        df = pd.merge(df_UI, df_IT, on='item')

        df['rate'] = 1
        df = df.dropna(axis=0, how='any')
        df.drop_duplicates(subset=['user', 'item'], keep='last', inplace=True)
        # df = self.remove_infrequent_items(df, item_filter)
        # df = self.remove_infrequent_users(df, user_filter)
        df = df.reset_index().drop(['index'], axis=1)

        df = df[df['tag'].apply(lambda x: len(set(x)) >= 5)]  # make sure each item has at least k tags
        df = self.remove_infrequent_items(df, item_filter)
        df = self.remove_infrequent_users(df, user_filter)
        # df = self.remove_infrequent_items(df, 5)
        # df = self.remove_infrequent_users(df, 5)

        # df, user_mapping = self.convert_unique_idx(df, 'user')
        # df, item_mapping = self.convert_unique_idx(df, 'item')
        # df, tag_mapping = self.convert_unique_idx(df, 'tag')
        print("start generate unique idx")
        # df, user_mapping = self.convert_unique_idx(df, 'user')
        # df, item_mapping = self.convert_unique_idx(df, 'item')
        df['user'] = df['user'].astype('category').cat.codes
        df['item'] = df['item'].astype('category').cat.codes
        df = df.sort_values(by=['user'])
        df = df.reset_index().drop(['index'], axis=1)
        # print("Complete assigning unique index to user and item:\n", df[['user', 'item', 'rate']])

        IT_df = generate_IT_matrix(df)
        IT_df['tag'] = IT_df['tag'].astype('category').cat.codes
        IT_df['rate'] = 1
        IT_df.drop_duplicates(subset=['item', 'tag'], keep='last', inplace=True)
        IT_df = IT_df.reset_index().drop(['index'], axis=1)
        # print("IT_df:", IT_df)

        self.num_user = len(df['user'].unique())
        self.num_item = len(df['item'].unique())
        self.num_tag = len(IT_df['tag'].unique())

        print('num_user', self.num_user)
        print('num_item', self.num_item)
        print('num_tag', self.num_tag)

        self.UI_matrix = scipy.sparse.csr_matrix(
            (np.array(df['rate']), (np.array(df['user']), np.array(df['item']))))

        self.IT_matrix = scipy.sparse.csr_matrix(
            (np.array(IT_df['rate']), (np.array(IT_df['item']), np.array(IT_df['tag']))))

        if val_ratio:
            val_ratio = val_ratio / (1 - test_ratio)

        self.train_matrix_UI, self.test_matrix_UI, self.val_matrix_UI, \
        self.train_set_UI, self.test_set_UI, self.val_set_UI = self.create_train_test_split(self.UI_matrix,
                                                                                            test_ratio=test_ratio,
                                                                                            val_ratio=val_ratio, seed=0)

        self.train_matrix_IT, self.test_matrix_IT, self.val_matrix_IT, \
        self.train_set_IT, self.test_set_IT, self.val_set_IT = self.create_train_test_split(self.IT_matrix,
                                                                                            test_ratio=test_ratio,
                                                                                            val_ratio=val_ratio, seed=0)

        print("U-I:")
        print(np.sum([len(x) for x in self.train_set_UI]),
              np.sum([len(x) for x in self.val_set_UI]),
              np.sum([len(x) for x in self.test_set_UI]))
        density = float(
            np.sum([len(x) for x in self.train_set_UI]) + np.sum([len(x) for x in self.val_set_UI]) + np.sum(
                [len(x) for x in self.test_set_UI])) / self.num_user / self.num_item
        print("density:{:.2%}".format(density))

        print("I-T:")
        print(np.sum([len(x) for x in self.train_set_IT]),
              np.sum([len(x) for x in self.val_set_IT]),
              np.sum([len(x) for x in self.test_set_IT]))
        density = float(
            np.sum([len(x) for x in self.train_set_IT]) + np.sum([len(x) for x in self.val_set_IT]) + np.sum(
                [len(x) for x in self.test_set_IT])) / self.num_tag / self.num_item
        print("density:{:.2%}".format(density))

        avg_deg = np.sum([len(x) for x in self.train_set_UI]) / self.num_user
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree U-I graph: ', avg_deg)

        avg_deg = np.sum([len(x) for x in self.train_set_IT]) / self.num_item
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree I-T graph: ', avg_deg)

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

    def create_train_test_split(self, rating_df, val_ratio=None, test_ratio=0.2, seed=0):
        data_set = []
        for item_ids in rating_df:
            data_set.append(item_ids.indices.tolist())
        # data_set = self.data_set

        train_set, test_set, val_set = self.split_data_randomly(data_set, val_ratio=val_ratio, test_ratio=test_ratio,
                                                                seed=seed)
        train_matrix = self.generate_rating_matrix(train_set, rating_df.shape[0], rating_df.shape[1])
        test_matrix = self.generate_rating_matrix(test_set, rating_df.shape[0], rating_df.shape[1])
        val_matrix = self.generate_rating_matrix(val_set, rating_df.shape[0], rating_df.shape[1])

        return train_matrix, test_matrix, val_matrix, train_set, test_set, val_set

    def generate_rating_matrix(self, train_matrix, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        # triplet = []
        for user_id, article_list in enumerate(train_matrix):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)
                # triplet.append((int(user_id), int(article), float(1)))

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

    def remove_infrequent_items(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['item'].value_counts()
        df = df[df['item'].isin(counts[counts >= min_counts].index)]

        print("items with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_users(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['user'].value_counts()
        df = df[df['user'].isin(counts[counts >= min_counts].index)]

        print("users with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def remove_infrequent_tags(self, data, min_counts=10):
        df = deepcopy(data)
        counts = df['tag'].value_counts()
        df = df[df['tag'].isin(counts[counts >= min_counts].index)]

        print("tags with < {} interactoins are removed".format(min_counts))
        # print(df.describe())
        return df

    def convert_unique_idx(self, df, column_name):
        column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
        df[column_name] = df[column_name].apply(column_dict.get)
        df[column_name] = df[column_name].astype('int')
        assert df[column_name].min() == 0
        assert df[column_name].max() == len(column_dict) - 1
        return df, column_dict


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--user_filter', type=int, default=10)
    parser.add_argument('--item_filter', type=int, default=10)
    parser.add_argument('--tag_filter', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parser_args()
    data_dir = '/home/haolun/projects/UIT/data/delicious'
    data_generator = Data(data_dir, test_ratio=parser.test_ratio, val_ratio=parser.val_ratio,
                          user_filter=parser.user_filter, item_filter=parser.item_filter, tag_filter=parser.tag_filter)

    # print("data_genrator:", data_generator)
