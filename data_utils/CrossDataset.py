import os
import torch
import pickle
import scipy.io as io
import numpy as np
from sklearn import preprocessing


def read_Train_dataset(data_folder='./data/', DATASET='CUB',DEVICE='cuda',attr_norm=True,att_type='clip'):
    DATA_DIR = os.path.join(data_folder, DATASET)
    if DATASET == "ImageNet":
        data_path = f'{DATA_DIR}/res101_useweight.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        att_path = f'{DATA_DIR}/clip_splits.pkl'
        with open(att_path, 'rb') as f:
            attrs_mat = pickle.load(f)
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze()  # Using "-1" here and for idx to normalize to 0-index
        train_idx = np.array(attrs_mat['trainval_loc'])
        test_seen_idx = np.array(attrs_mat['test_seen_loc'])
        test_unseen_idx = np.array(attrs_mat['test_unseen_loc'])
    else:
        data = io.loadmat(f'{DATA_DIR}/res101_useweight.mat')

        if att_type=='w2v':
            attrs_mat = io.loadmat(f'{DATA_DIR}/w2v_wiki_en_splits.mat')
        else:
            attrs_mat = io.loadmat('{}/{}_splits.mat'.format(DATA_DIR,att_type))
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1  # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
        test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

    test_idx = np.array(test_seen_idx.tolist() + test_unseen_idx.tolist())

    seen_classes = sorted(np.unique(labels[test_seen_idx]))
    unseen_classes = sorted(np.unique(labels[test_unseen_idx]))

    print(f'<=============== Preprocessing ===============>')
    num_classes = len(seen_classes) + len(unseen_classes)
    seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
    unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
    attrs = attrs_mat['att'].T
    # attrs = torch.from_numpy(attrs).to(DEVICE).float()
    attrs = torch.from_numpy(attrs).to(DEVICE).float()
    if attr_norm:
        attrs = attrs / attrs.norm(dim=1, keepdim=True) * np.sqrt(attrs.shape[1])

    attrs_seen = attrs[seen_mask]
    attrs_unseen = attrs[unseen_mask]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    test_feats = feats[test_idx]
    test_seen_idx = [i for i, y in enumerate(test_labels) if y in seen_classes]
    test_unseen_idx = [i for i, y in enumerate(test_labels) if y in unseen_classes]

    labels_remapped_to_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in labels]
    test_labels_remapped_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in test_labels]
    test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in test_labels]
    ds_train = [(feats[i], labels_remapped_to_seen[i]) for i in train_idx]
    ds_test_seen = [(test_feats[i], test_labels_remapped_seen[i]) for i in test_seen_idx]
    ds_test_unseen = [(test_feats[i], test_labels_remapped_unseen[i]) for i in test_unseen_idx]

    return ds_train, ds_test_seen, ds_test_unseen, attrs, attrs_seen, attrs_unseen, seen_mask


def read_Test_dataset(data_folder='./data/', DATASET='CUB',DEVICE='cuda',attr_norm=True, att_type = 'clip'):
    DATA_DIR = os.path.join(data_folder,DATASET)
    if DATASET == "ImageNet":
        data_path = f'{DATA_DIR}/res101_useweight.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        att_path = f'{DATA_DIR}/clip_splits.pkl'
        with open(att_path, 'rb') as f:
            attrs_mat = pickle.load(f)
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze()  # Using "-1" here and for idx to normalize to 0-index
        attrs = attrs_mat['att'].T
        attrs = torch.from_numpy(attrs).to(DEVICE).float()
        return feats, labels, attrs
    else:
        data = io.loadmat(f'{DATA_DIR}/res101_useweight.mat')

        if att_type == 'w2v':
            attrs_mat = io.loadmat(f'{DATA_DIR}/w2v_wiki_en_splits.mat')
        else:
            attrs_mat = io.loadmat('{}/{}_splits.mat'.format(DATA_DIR,att_type))
        feats = data['features'].T.astype(np.float32)
        labels = data['labels'].squeeze() - 1  # Using "-1" here and for idx to normalize to 0-index
        attrs = attrs_mat['att'].T
        attrs = torch.from_numpy(attrs).to(DEVICE).float()
        if attr_norm:
            attrs = attrs / attrs.norm(dim=1, keepdim=True) * np.sqrt(attrs.shape[1])

        test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
        test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1
        seen_classes = sorted(np.unique(labels[test_seen_idx]))
        unseen_classes = sorted(np.unique(labels[test_unseen_idx]))
        num_classes = len(seen_classes) + len(unseen_classes)
        unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
        attrs_unseen = attrs[unseen_mask]
        unseen_feats = feats[test_unseen_idx]
        unseen_labels = labels[test_unseen_idx]
        test_labels_remapped_unseen = np.array([(unseen_classes.index(t) if t in unseen_classes else -1) for t in unseen_labels])
        return unseen_feats, test_labels_remapped_unseen, attrs_unseen


def GetCrossDataset(data_folder='./data/', train_dataset='CUB',test_mode='similar',att_type='clip'):
    AllDatasets=['CUB','AWA2','SUN']
    if test_mode=="similar":
        test_dataset_list = [train_dataset]
    elif test_mode=='dissimilar':
        test_dataset_list = [dataset for dataset in AllDatasets if dataset!=train_dataset]
    elif test_mode=='similar&dissimilar':
        test_dataset_list = AllDatasets
    print('P:{}|T:{}'.format(train_dataset,test_dataset_list))
    ds_P, ds_P_seen, ds_P_unseen, P_attrs, P_attrs_seen, P_attrs_unseen, P_seen_mask = read_Train_dataset(data_folder=data_folder, DATASET=train_dataset,att_type=att_type)

    # Test seen data
    feats_list = [np.expand_dims(ds_P_seen[i][0], axis=0) for i in range(len(ds_P_seen))]
    feats_item = np.concatenate(feats_list, axis=0)
    labels_list = [ds_P_seen[i][1] for i in range(len(ds_P_seen))]
    labels_item = np.array(labels_list)
    attrs_item = P_attrs_seen

    T_attrs = attrs_item
    T_feats = feats_item
    T_labels = labels_item
    unseen_start_id = P_attrs_seen.shape[0]
    for i,test_dataset in enumerate(test_dataset_list):
        if test_dataset == train_dataset:
            feats_list = [np.expand_dims(ds_P_unseen[i][0],axis=0) for i in range(len(ds_P_unseen))]
            feats_item = np.concatenate(feats_list,axis=0)
            labels_list = [ds_P_unseen[i][1] for i in range(len(ds_P_unseen))]
            labels_item = np.array(labels_list)
            attrs_item = P_attrs_unseen

        else:
            feats_item, labels_item, attrs_item = read_Test_dataset(data_folder=data_folder, DATASET=test_dataset,att_type=att_type)
        labels_item = labels_item + unseen_start_id


        T_attrs = torch.cat([T_attrs,attrs_item],dim=0)
        T_feats = np.concatenate([T_feats,feats_item],axis=0)
        T_labels = np.concatenate([T_labels,labels_item],axis=0)

        unseen_start_id = unseen_start_id + attrs_item.shape[0]

    ds_T = [(T_feats[i], T_labels[i]) for i in range(T_labels.shape[0])]

    return ds_P, ds_T, P_attrs_seen, T_attrs, T_labels

