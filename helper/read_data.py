import os
import sys

[sys.path.append(i) for i in ['.', '..']]
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import argparse
import scipy
import numpy as np
from scipy.sparse import csr_matrix
from copy import deepcopy
import pandas as pd

import torch


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class Read_data(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = data_dir

    def load(self):
        # Load data
        if "amazon" in self.fpath:
            df = pd.read_csv(self.fpath,
                             sep=',',
                             engine='python',
                             names=['user_id', 'item_id', 'rating', 'timestamp'])
            df = df.sort_values(by='user_id').reset_index().drop(['index'], axis=1)
            df = df[['user_id', 'item_id', 'rating']].rename(columns={"user_id": "user", "item_id": "item"})
        elif "modcloth" in self.fpath:
            df = pd.read_json(self.fpath, lines=True)
            df = df.sort_values(by='user_id').reset_index().drop(['index'], axis=1)
            df = df[['user_id', 'item_id', 'quality']].rename(
                columns={"user_id": "user", "item_id": "item", "quality": "rating"})
        return df


class ModCloth(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = data_dir

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep=',',
                              engine='python',
                              skiprows=[0],
                              names=['item_id', 'user_id', 'rating', 'timestamp', 'size', 'fit', 'user_attr',
                                     'model_attr', 'category', 'brand', 'year', 'split'],
                              usecols=['user_id', 'item_id', 'rating']).rename(
            columns={"user_id": "user", "item_id": "item", "rate": "rating"})
        print(df_rate)

        return df_rate


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='::',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate']).rename(columns={"rate": "rating"})

        return df_rate


class MovieLens100k(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'u.data')
        self.fpath_genre = os.path.join(data_dir, 'u.item')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='\t',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate']).rename(columns={"rate": "rating"})

        df_genre = pd.read_csv(self.fpath_genre,
                               sep='|',
                               engine='python',
                               names=['item', 'title', 'date1', 'date2', 'url', 'g0', 'g1', 'g2', 'g3', 'g4', 'g5',
                                      'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15', 'g16', 'g17',
                                      'g18'],
                               usecols=['item', 'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10',
                                        'g11', 'g12', 'g13', 'g14', 'g15', 'g16', 'g17', 'g18']
                               )

        df_rate = pd.merge(df_rate, df_genre, on='item')

        return df_rate


def genre_ml100k_index(df):
    df_genre = df[
        ['item', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15',
         'g16', 'g17', 'g18']]
    df_genre = df_genre.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True).drop(columns=['item'])
    genre_name = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15',
                  'g16', 'g17', 'g18']
    index_genre = []
    for genre in genre_name:
        index_genre.append(torch.tensor(np.flatnonzero(df_genre[genre])).long())

    genre_mask = df_genre.to_numpy().T
    genre_mask = torch.FloatTensor(genre_mask)

    return index_genre, genre_mask


class LastFM(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'usersha1-artmbid-artname-plays.tsv')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='\t',
                              names=['user', 'item', 'artist', 'rate'],
                              usecols=['user', 'item', 'rate']
                              ).rename(columns={"rate": "rating"})
        df_rate['user'] = df_rate['user'].astype(str)

        df_rate = df_rate.dropna(axis=0, how='any')

        return df_rate


class Yelp(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'yelp_academic_dataset_review.json')

    def load(self):
        # Load data
        # with open(self.fpath_rate) as f:
        #     data = json.load(f)
        # df = pd.DataFrame(data)
        df = pd.read_json(self.fpath_rate, lines=True)
        df = df.sort_values(by='user_id').reset_index().drop(['index'], axis=1)
        df = df[['user_id', 'business_id', 'stars']].rename(
            columns={"user_id": "user", "business_id": "item", "stars": "rating"})

        df_rate = df.dropna(axis=0, how='any')

        return df_rate


class Data(object):
    def __init__(self, data_dir, data_name, val_ratio=None, test_ratio=0.2, user_filter=5, item_filter=5, clean='N',
                 seed=0):

        file_path = os.path.join("../preprocessed/",
                                 '{}_{}_{}.pkl'.format(data_name, str(user_filter), str(item_filter)))
        #
        if os.path.exists(file_path):
            df = pd.read_pickle(file_path)
            # print("load saved:\n", df)
        else:
            if data_name == 'mv':
                df = MovieLens1M(data_dir).load()
            if data_name == 'mv100':
                df = MovieLens100k(data_dir).load()
            elif data_name == 'm':
                df = ModCloth(data_dir).load()
            elif data_name == 'l':
                df = LastFM(data_dir).load()
            elif data_name == 'y':
                df = Yelp(data_dir).load()
            else:
                df = Read_data(data_dir).load()
            # print("finish loading raw data:", df)

            df.drop_duplicates(subset=['user', 'item'], keep='last', inplace=True)
            df = self.remove_infrequent_items(df, item_filter)
            df = self.remove_infrequent_users(df, user_filter)
            df = df.reset_index().drop(['index'], axis=1)
            # print("remove infrequent users and itemds:", df)

            print("start generate unique idx")
            df['user'] = df['user'].astype('category').cat.codes
            df['item'] = df['item'].astype('category').cat.codes
            df = df.reset_index().drop(['index'], axis=1)
            df['implicit'] = df['rating']

            df.loc[df['implicit'] > 0, 'implicit'] = 1
            print("Complete assigning unique index to user and item:\n", df)

            df.to_pickle(file_path)

        if data_name == 'mv100':
            self.index_genre, self.genre_mask = genre_ml100k_index(df)

        df_real = df.loc[df['rating'] > 3.0]
        df_noise = df.loc[df['rating'] <= 3.0]
        df_real = df_real.reset_index().drop(['index'], axis=1)
        print("make sure each user has 5 real preference ratings...")
        df_real = self.remove_infrequent_users(df_real, 5)
        df_noise = df_noise[df_noise['user'].isin(df_real['user'])]

        # if data_name == 'm':
        if data_name in ['b', 'e', 'l', 'y']:
            df_real = df_real[df_real['user'].isin(df_noise['user'])]
            # df_noise = df_noise[df_noise['item'].isin(df_real['item'])]
            df_real = df_real[df_real['item'].isin(df_noise['item'])]
            df_real = self.remove_infrequent_users(df_real, 5)
            df_noise = df_noise[df_noise['user'].isin(df_real['user'])]

        df_real_noise = pd.concat([df_real, df_noise], ignore_index=True)
        df_real_noise['user'] = df_real_noise['user'].astype('category').cat.codes
        df_real_noise['item'] = df_real_noise['item'].astype('category').cat.codes
        df_real_noise = df_real_noise.reset_index().drop(['index'], axis=1)
        # print("df_real_noise:\n", df_real_noise)

        df_real_noise.to_pickle(
            os.path.join("../preprocessed/", '{}_final.pkl'.format(data_name)))

        self.num_user = len(df_real_noise['user'].unique())
        self.num_item = len(df_real_noise['item'].unique())

        print('num_user', self.num_user)
        print('num_item', self.num_item)

        # re-construct
        df_real = df_real_noise.loc[df_real_noise['rating'] > 3.0]
        df_noise = df_real_noise.loc[df_real_noise['rating'] <= 3.0]

        self.matrix_real = scipy.sparse.csr_matrix(
            (np.array(df_real['implicit']), (np.array(df_real['user']), np.array(df_real['item']))))
        self.matrix_noise = scipy.sparse.csr_matrix(
            (np.array(df_noise['implicit']), (np.array(df_noise['user']), np.array(df_noise['item']))))

        print("self.matrix_real:", self.matrix_real.shape)
        print("self.matrix_noise:", self.matrix_noise.shape)

        if val_ratio:
            val_ratio = val_ratio / (1 - test_ratio)

        self.train_matrix_real, self.test_matrix_real, self.val_matrix_real, \
        self.train_set_real, self.test_set_real, self.val_set_real \
            = self.create_train_test_split(self.matrix_real, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)

        self.test_matrix = self.test_matrix_real
        self.val_matrix = self.val_matrix_real
        self.test_set = self.test_set_real
        self.val_set = self.val_set_real

        if clean == 'N':
            self.train_matrix = self.train_matrix_real + self.matrix_noise
            self.train_set_noise = self.construct_train_set(self.matrix_noise)
            self.train_set = [i + j for i, j in zip(self.train_set_real, self.train_set_noise)]
        elif clean == 'Y':
            self.train_matrix = self.train_matrix_real
            self.train_set = self.train_set_real
        # print("self.train_set:", len(self.train_set), self.train_set[:10])

        print("U-I:")
        print(np.sum([len(x) for x in self.train_set]),
              np.sum([len(x) for x in self.val_set]),
              np.sum([len(x) for x in self.test_set]))
        density = float(
            np.sum([len(x) for x in self.train_set]) + np.sum([len(x) for x in self.val_set]) + np.sum(
                [len(x) for x in self.test_set])) / self.num_user / self.num_item
        print("density:{:.3%}".format(density))

        avg_deg = np.sum([len(x) for x in self.train_set]) / self.num_user
        avg_deg *= 1 / (1 - test_ratio)
        print('Avg. degree U-I graph: ', avg_deg)

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

    def construct_train_set(self, rating_df, seed=0):
        user_records = []
        for item_ids in rating_df:
            user_records.append(item_ids.indices.tolist())

        train_set = []
        for user_id, item_list in enumerate(user_records):
            train_sample = []
            for place in item_list:
                train_sample.append(place)

            train_set.append(train_sample)

        return train_set

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


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_name", type=str, default="c",
                        choices=['b', 'c', 'm', 'mv', 'mv100', 'e', 'l', 'y'])
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--user_filter', type=int, default=10)
    parser.add_argument('--item_filter', type=int, default=10)
    parser.add_argument('--clean', type=str, default='N', choices=['Y', 'N'])
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    if args.data_name == 'b':
        data_dir = '../data/amazon-beauty/ratings_Beauty.csv'
        args.user_filter = 10
        args.item_filter = 15
    elif args.data_name == 'c':
        data_dir = '../data/amazon-cd/ratings_CDs_and_Vinyl.csv'
        args.user_filter = 15
        args.item_filter = 20
    elif args.data_name == 'm':
        data_dir = '../data/modcloth/df_modcloth.csv'
        args.user_filter = 1
        args.item_filter = 1
    elif args.data_name == 'mv':
        data_dir = '../data/ml-1m'
        args.user_filter = 10
        args.item_filter = 10
    elif args.data_name == 'mv100':
        data_dir = '../data/ml-100k'
        args.user_filter = 10
        args.item_filter = 10
    elif args.data_name == 'e':
        data_dir = '../data/amazon-electronics/ratings_Electronics.csv'
        args.user_filter = 10
        args.item_filter = 10
    elif args.data_name == 'l':
        data_dir = '../data/lastfm/'
        args.user_filter = 20
        args.item_filter = 20
    elif args.data_name == 'y':
        data_dir = '../data/yelp_dataset/'
        args.user_filter = 30
        args.item_filter = 30
    # elif args.data_name == 'e':
    #     data_dir = '../data/amazon-electronics/ratings_Electronics.csv'
    #     args.user_filter = 10
    #     args.item_filter = 10

    dataset = args.data_name

    data_generator = Data(data_dir, dataset, test_ratio=args.test_ratio, val_ratio=args.val_ratio,
                          user_filter=args.user_filter, item_filter=args.item_filter, clean=args.clean, seed=args.seed)
