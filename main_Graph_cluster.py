import sys
import logging
import numpy as np
import torch
import time

from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from helper.read_data import Data
from helper import read_data_LIST
from argparse import ArgumentParser
from model.Graph_cluster import NGCF, Controller
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from kmeans_pytorch import kmeans

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser(description="Adaptive Margin for CF")
    parser.add_argument("--data_name", type=str, default="c", choices=['b', 'c', 'e', 'g', 'l', 'y'])
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--clean', type=str, default='N', choices=['Y', 'N'])
    parser.add_argument('--user_filter', type=int, default=10)
    parser.add_argument('--item_filter', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--is_logging', type=bool, default=False)
    # neighborhood to use
    parser.add_argument('--cluster', type=int, default=20, help="number of clustering")
    parser.add_argument('--n_neg', type=int, default=1, help='the number of negative samples')
    # Seed
    parser.add_argument('--seed', type=int, default=3, help="Seed")
    # Model
    parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
    parser.add_argument('--bl', type=str, default='Y', choices=['N', 'Y'], help="whether to use bilevel optimization")
    parser.add_argument('--trick', type=str, default='raw', choices=['norm', 'raw', 'clip'])
    parser.add_argument('--norm', type=str, default='N', choices=['N', 'Y'])
    parser.add_argument('--state', type=int, default=19)

    # model
    parser.add_argument('--model', type=str, default='l', choices=['n', 'g', 'm', 'l'])
    parser.add_argument('--n_fold', type=int, default=20)
    parser.add_argument('--layer_size', nargs='?', default='[64]', help='Output sizes of every layer')
    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    # parser.add_argumednt('-l', '--list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]', help="keep 1 - x")
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]', help="keep 1 - x")

    # Optimizer
    parser.add_argument('--lr1', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd1', type=float, default=1e-3, help="Weight decay factor")
    parser.add_argument('--lr2', type=float, default=1e-3, help="Learning rate on k")
    parser.add_argument('--wd2', type=float, default=0, help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs', type=int, default=1000, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=50,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--batch_size', type=int, default=5000, help="batch_size")
    parser.add_argument('--topk', type=int, default=20, help="topk")
    parser.add_argument('--sample_every', type=int, default=10, help="sample frequency")

    return parser.parse_args()


def neg_item_pre_sampling(train_matrix, num_neg_candidates=500):
    num_users, num_items = train_matrix.shape
    user_neg_items = []
    for user_id in range(num_users):
        pos_items = train_matrix[user_id].indices

        u_neg_item = negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates)
        user_neg_items.append(u_neg_item)

    user_neg_items = np.asarray(user_neg_items)

    return user_neg_items


def generate_pred_list(model, train_matrix, topk=20):
    num_users = train_matrix.shape[0]
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    pred_list = None

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(args.device)

        rating_pred = model.predict(batch_user_ids)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    return pred_list


def compute_metrics(test_set, pred_list, topk=20):
    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg


def max_norm(param, max_val=1, eps=1e-8):
    norm = param.norm(2, dim=1, keepdim=True)
    desired = torch.clamp(norm, 0, max_val)
    param = param * (desired / (eps + norm))

    return param


def generate_controller_state(user, pos, neg, pos_center, neg_center, uni_center, norm, state):
    if norm == 'Y':
        user = max_norm(user)
        pos = max_norm(pos)
        neg = max_norm(neg)
        pos_center = max_norm(pos_center)
        neg_center = max_norm(neg_center)
    A = torch.pow(user - pos, 2)  # u-i
    a = torch.pow(user - neg, 2)  # u-j
    B = torch.pow(user - pos_center, 2)  # u-c1
    b = torch.pow(user - neg_center, 2)  # u-c4
    C = torch.pow(pos_center - pos, 2)  # i-c1
    c = torch.pow(neg_center - neg, 2)  # j-c4

    A_pdt = user * pos  # u*i
    a_pdt = user * neg  # u*j
    B_pdt = user * pos_center  # u*c1
    b_pdt = user * neg_center  # u*c4
    C_pdt = pos_center * pos  # i*c1
    c_pdt = neg_center * neg  # j*c4

    if args.state == 0:
        return torch.cat((
            A + B + C, a + b + c, (a + b + c) - (A + B + C),
        ), dim=1)
    elif args.state == 1:
        return torch.cat((
            A + B + C, a + b + c,
        ), dim=1)
    elif args.state == 2:
        return torch.cat((
            A, B, C, a, b, c,
        ), dim=1)
    elif args.state == 3:
        return torch.cat((
            A_pdt + B_pdt + C_pdt, a_pdt + b_pdt + c_pdt, (a_pdt + b_pdt + c_pdt) - (A_pdt + B_pdt + C_pdt),
        ), dim=1)
    elif args.state == 4:
        return torch.cat((
            A_pdt + B_pdt + C_pdt, a_pdt + b_pdt + c_pdt,
        ), dim=1)
    elif args.state == 5:
        return torch.cat((
            A_pdt, B_pdt, C_pdt, a_pdt, b_pdt, c_pdt,
        ), dim=1)
    elif args.state == 6:
        return torch.cat((
            A, B, C, (A + B + C),
            a, b, c, (a + b + c),
        ), dim=1)
    elif args.state == 7:
        return torch.cat((
            A_pdt, B_pdt, C_pdt, (A_pdt + B_pdt + C_pdt),
            a_pdt, b_pdt, c_pdt, (a_pdt + b_pdt + c_pdt),
        ), dim=1)
    elif args.state == 8:
        return torch.cat((
            A, a, a - A,
            B, b, b - B,
            C, c, c - C
        ), dim=1)
    elif args.state == 9:
        return torch.cat((
            A_pdt, a_pdt, a_pdt - A_pdt,
            B_pdt, b_pdt, b_pdt - B_pdt,
            C_pdt, c_pdt, c_pdt - C_pdt
        ), dim=1)


def train_model(args):
    # pre-sample a small set of negative samples
    t1 = time.time()
    user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
    pre_samples = {'user_neg_items': user_neg_items}

    print("Pre sampling time:{}".format(time.time() - t1))

    model = NGCF(user_size, item_size, args).to(args.device)
    len_list = [3, 2, 6, 3, 2, 6, 8, 8, 9, 9]
    state_length = len_list[args.state]
    controller = Controller(state_length * args.dim, args.device).to(args.device)

    model_optimizer = torch.optim.Adam(model.myparameters, lr=args.lr1, weight_decay=args.wd1)
    weight_optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr2, weight_decay=args.wd2)

    sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.n_neg, n_workers=4)
    num_batches = train_matrix.count_nonzero() // args.batch_size
    print("num_batches:", num_batches)

    save_weight_min_max = [[], []]

    try:
        for iter in range(1, args.n_epochs + 1):
            # logger.info("Epochs:{}".format(iter + 1))
            check_iter_condition = (iter >= 10)

            start = time.time()
            model.train()

            loss = 0.
            avg_in_cost = 0.
            avg_out_cost = 0.

            """use k-means every 10 epochs"""
            if check_iter_condition:
                """Use the k-means clustering label"""
                if iter % 10 == 0:
                    num_clusters = args.cluster
                    cluster_ids, cluster_centers = kmeans(
                        X=model.item_embeddings.weight.data, num_clusters=num_clusters, distance='euclidean',
                        tqdm_flag=False, device=args.device
                    )
                    cluster_ids = cluster_ids.to(args.device)  # -1 class for pad

            # Start Training
            for batch_id in range(num_batches):

                # get mini-batch data
                batch_user_id, batch_item_id, neg_samples = sampler.next_batch()
                user, pos, neg = batch_user_id, batch_item_id, np.squeeze(neg_samples)

                if args.bl == 'N' or check_iter_condition == False:
                    user_emb, pos_emb, neg_emb = model.forward_unify(user, pos, neg)

                    batch_loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
                    batch_loss = torch.sum(batch_loss)

                    model_optimizer.zero_grad()
                    weight_optimizer.zero_grad()
                    batch_loss.backward()
                    model_optimizer.step()
                    weight_optimizer.step()

                    loss += batch_loss.item()

                elif args.bl == 'Y' and check_iter_condition == True:
                    user_emb_ego, pos_emb_ego, neg_emb_ego, user_emb, pos_emb, neg_emb, pos_center, neg_center, uni_center = \
                        model.forward_person_plus(user, pos, neg, cluster_ids)

                    state = generate_controller_state(user_emb_ego, pos_emb_ego, neg_emb_ego, pos_center, neg_center,
                                                      uni_center,
                                                      args.norm, args.state)

                    loss_weight = controller(state)
                    if args.trick == 'norm':
                        loss_weight = loss_weight / float(loss_weight.sum()) * args.batch_size
                    elif args.trick == 'clip':
                        loss_weight = torch.clamp(loss_weight, 0, 5)
                    elif args.trick == 'raw':
                        pass
                    in_loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
                    in_loss = torch.sum(in_loss * loss_weight)

                    # make a copy for embeddings
                    user_embeddings = model.user_embeddings.weight.clone()
                    item_embeddings = model.item_embeddings.weight.clone()
                    # weight_embeddings = {k: model.weight_dict[k].clone() for k in model.weight_dict}

                    model_optimizer.zero_grad()
                    grad_users = torch.autograd.grad(in_loss, model.user_embeddings.weight, create_graph=True)[0]
                    grad_items = torch.autograd.grad(in_loss, model.item_embeddings.weight, create_graph=True)[0]
                    # grad_weights = {k: torch.autograd.grad(in_loss, model.weight_dict[k], create_graph=True)[0]
                    #                 for k in model.weight_dict}
                    in_loss.backward(retain_graph=True)

                    # update ids
                    updated_user_ids = np.array(list(set(user)))
                    updated_user_ids = torch.from_numpy(updated_user_ids).type(torch.LongTensor).to(args.device)
                    updated_item_ids = np.array(list(set(pos).union(set(neg.flatten()))))
                    updated_item_ids = torch.from_numpy(updated_item_ids).type(torch.LongTensor).to(args.device)

                    # Compute gradient on Theta
                    grad_users = grad_users[updated_user_ids]
                    grad_items = grad_items[updated_item_ids]

                    # SGD update
                    alpha = 1e-3
                    user_embeddings[updated_user_ids] = user_embeddings[updated_user_ids] - alpha * grad_users
                    item_embeddings[updated_item_ids] = item_embeddings[updated_item_ids] - alpha * grad_items
                    # weight_embeddings = {k: weight_embeddings[k] - alpha * grad_weights[k]
                    #                      for k in model.weight_dict}

                    # use the new threshold_u to update user_emb
                    user_emb, pos_emb, neg_emb = model.out_forward(user, pos, neg,
                                                                   user_embeddings, item_embeddings)

                    out_loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
                    out_loss = torch.sum(out_loss)

                    weight_optimizer.zero_grad()
                    model.user_embeddings.requires_grad = False
                    model.item_embeddings.requires_grad = False
                    # for k in grad_weights:
                    #     model.weight_dict[k].requires_grad = False
                    out_loss.backward()
                    model.user_embeddings.requires_grad = True
                    model.item_embeddings.requires_grad = True
                    # for k in grad_weights:
                    #     model.weight_dict[k].requires_grad = True
                    weight_optimizer.step()
                    model_optimizer.step()

                    grad_users.detach_()
                    grad_items.detach_()
                    # for k in grad_weights:
                    #     grad_weights[k].detach_()

                    avg_in_cost += in_loss / num_batches
                    avg_out_cost += out_loss / num_batches

            logger.info("Epochs:{}".format(iter))
            if args.bl == 'N' or check_iter_condition == False:
                logger.info('Avg BPR loss:{:.6f}'.format(loss / num_batches))
            elif args.bl == 'Y' and check_iter_condition == True:
                logger.info("[{:.2f}s] Avg in_loss:{:.6f}, Avg out_loss:{:.6f}".format(time.time() - start, avg_in_cost,
                                                                                       avg_out_cost))
                logger.info(
                    'loss_weight:max:{:.6f}, min:{:.6f}'.format(max(loss_weight).item(), min(loss_weight).item()))

            if iter % args.every == 0:
                logger.info("Epochs:{}".format(iter))
                if args.bl == 'N' or check_iter_condition == False:
                    logger.info('Avg BPR loss:{:.6f}'.format(loss / num_batches))
                elif args.bl == 'Y' and check_iter_condition == True:
                    logger.info(
                        "[{:.2f}s] Avg in_loss:{:.6f}, Avg out_loss:{:.6f}".format(time.time() - start, avg_in_cost,
                                                                                   avg_out_cost))
                    logger.info(
                        'loss_weight:max:{:.6f}, min:{:.6f}'.format(max(loss_weight).item(), min(loss_weight).item()))
                model.eval()
                pred_list = generate_pred_list(model, train_matrix, topk=20)
                if args.val_ratio > 0.0:
                    print("validation:")
                    precision, recall, MAP, ndcg = compute_metrics(val_user_list, pred_list, topk=20)
                    logger.info(', '.join(str(e) for e in recall))
                    logger.info(', '.join(str(e) for e in ndcg))

                print("test:")
                precision, recall, MAP, ndcg = compute_metrics(test_user_list, pred_list, topk=20)
                logger.info(', '.join(str(e) for e in recall))
                logger.info(', '.join(str(e) for e in ndcg))

            if iter % args.sample_every == 0:
                user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
                pre_samples = {'user_neg_items': user_neg_items}
                sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.n_neg,
                                     n_workers=4)

        sampler.close()
    except KeyboardInterrupt:
        sampler.close()
        sys.exit()

    logger.info('Parameters:')
    for arg, value in sorted(vars(args).items()):
        logger.info("%s: %r", arg, value)
    logger.info('\n')
    print("Whole time:{}".format(time.time() - t1))


if __name__ == '__main__':
    args = parse_args()
    if args.data_name == 'c':
        data_dir = './data/amazon-cd/ratings_CDs_and_Vinyl.csv'
        args.user_filter = 15
        args.item_filter = 20
        args.batch_size = 5000
    elif args.data_name == 'g':
        data_dir = './data/gowalla/'
        args.user_filter = 2
        args.item_filter = 1
        args.batch_size = 5000
    elif args.data_name == 'l':
        data_dir = './data/lastfm/'
        args.user_filter = 20
        args.item_filter = 20
        args.batch_size = 5000

    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)

    if args.is_logging is True:
        handler = logging.FileHandler('./log/' + args.data + '.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    print(args)

    print("Data name:", args.data_name)
    if args.data_name == 'g':
        data_generator = read_data_LIST.Data(dataset="gowalla", pkl_path=data_dir, test_ratio=args.test_ratio,
                                             val_ratio=args.val_ratio, seed=args.seed)
    else:
        data_generator = Data(data_dir, args.data_name, test_ratio=args.test_ratio, val_ratio=args.val_ratio,
                              user_filter=args.user_filter, item_filter=args.item_filter, clean=args.clean,
                              seed=args.seed)

    user_size, item_size = data_generator.num_user, data_generator.num_item
    train_user_list, val_user_list, test_user_list = data_generator.train_set, data_generator.val_set, data_generator.test_set
    val_user_list = np.array(val_user_list, dtype=list)
    test_user_list = np.array(test_user_list, dtype=list)

    train_matrix = data_generator.train_matrix
    val_matrix = data_generator.val_matrix
    test_matrix = data_generator.test_matrix

    R = train_matrix.todok()
    args.R = R

    train_label = torch.from_numpy(train_matrix.todense()).float().to(args.device)
    args.train_label = train_label

    len_list = sorted(np.array(list(map(lambda x: len(x), train_user_list))))
    print("top_max_len:", len_list[-10:])

    train_model(args)
