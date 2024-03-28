import os
# os.environ['CUDA_VISIBLE_DEVICES']='5'
import sys
import argparse
import torch
import numpy as np
from time import time
from tqdm import tqdm
from scipy import io
from torch.utils.data import DataLoader
from models.DSECN import DSECN
from data_utils.CrossDataset import GetCrossDataset


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=10) #CUB-w2v:10 ,AWA2:58, SUN:58
    # parser.add_argument('--att_type',type=str,default='clip')
    parser.add_argument('--att_type', type=str, default='w2v')
    parser.add_argument('--train_dataset',type=str,default='CUB')
    parser.add_argument('--ECN_type', type=str, default='1k')
    parser.add_argument('--use_data',action='store_true',default=True)
    parser.add_argument('--use_DSE', action='store_true', default=True)
    # parser.add_argument('--use_HTE', action='store_true', default=True)
    # parser.add_argument('--use_DSE', action='store_true', default=False)
    parser.add_argument('--use_HTE', action='store_true', default=False)
    args = parser.parse_args()
    return args


def train(args):
    ## Train Model
    run_folder = os.getcwd()
    data_folder = os.path.join(run_folder, 'data')
    seed = args.seed
    att_type = args.att_type
    train_dataset = args.train_dataset
    ECN_type = args.ECN_type

    EPOCHS = 50
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'<=============== Loading train data for {train_dataset} ===============>')
    ds_train,ds_test,P_attrs_seen, T_attrs, T_labels = GetCrossDataset(data_folder=data_folder, train_dataset=train_dataset,test_mode='similar',att_type=att_type)
    DEVICE = 'cuda'
    num_classes = T_attrs.shape[0]
    num_seenclasses = P_attrs_seen.shape[0]
    seen_classes = sorted([i for i in range(num_seenclasses)])
    unseen_classes = sorted([i + num_seenclasses for i in range(num_classes - num_seenclasses)])
    seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
    unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
    train_dataloader = DataLoader(ds_train, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=2048)

    print(f'<=============== Loading semantics data from External base class names ===============>')
    if att_type=='clip':
        imagenet_clip_path = "{}/data/ExternalClass/clip_splits_1k.npy".format(run_folder)
        clip_1k = np.load(imagenet_clip_path)
        label_1k = np.array([i for i in range(clip_1k.shape[0])])

    elif att_type=='w2v':
        imagenet_clip_path = "{}/data/ExternalClass/w2v_wiki_en_splits_1k.npy".format(run_folder)
        clip_1k = np.load(imagenet_clip_path)
        label_1k = np.array([i for i in range(clip_1k.shape[0])])

    clip_1k = torch.from_numpy(clip_1k).float().to(DEVICE)
    label_1k = torch.from_numpy(label_1k).to(DEVICE)
    print(f'<=============== Loading semantics data from hierarchy taxonomy class names ===============>')
    if att_type=='clip':
        hypo_path = "{}/data/ExternalClass/clip_splits_hypoclass.npy".format(run_folder)

    elif att_type=='w2v':
        hypo_path = "{}/data/ExternalClass/w2v_wiki_en_splits_hypoclass.npy".format(run_folder)

    hypo_data = np.load(hypo_path,allow_pickle=True).item()
    clip_hypo = hypo_data['feature']
    label_hypo = np.array(hypo_data['target'])
    clip_hypo = torch.from_numpy(clip_hypo).float().to(DEVICE)
    label_hypo = torch.from_numpy(label_hypo).to(DEVICE)

    print(f'\n<=============== Starting training ===============>')
    start_time = time()
    model = DSECN(T_attrs.shape[1], 1024, 2048, args=args).to(DEVICE)
    print(model)

    optim = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    for epoch in tqdm(range(EPOCHS)):
        model.train()

        for i, batch in enumerate(train_dataloader):
            feats = torch.from_numpy(np.array(batch[0])).to(DEVICE)
            targets = torch.from_numpy(np.array(batch[1])).to(DEVICE)
            loss = model.train_loss(T_attrs[seen_mask], feats, targets, clip_1k, label_1k, clip_hypo, label_hypo)
            optim.zero_grad()
            loss.backward()
            optim.step()


    print(f'Training is done! Took time: {(time() - start_time): .1f} seconds')


    attrs = T_attrs
    test_labels = T_labels
    class_indices_inside_test = {c: [i for i in range(len(T_labels)) if T_labels[i] == c] for c in
                                 range(num_classes)}
    test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in T_labels]
    model.eval()  # Important! Otherwise we would use unseen batch statistics
    logits = [model(x.to(DEVICE), attrs).cpu() for x, _ in test_dataloader]
    logits = torch.cat(logits, dim=0)
    if att_type=='w2v' and train_dataset=='CUB':
        logits[:, seen_mask] *= 0.95
    else:
        logits[:, seen_mask] *= (0.95 if train_dataset != "CUB" else 1.0)  # Trading a bit of gzsl-s for a bit of gzsl-u

    preds_gzsl = logits.argmax(dim=1).numpy()
    preds_zsl_s = logits[:, seen_mask].argmax(dim=1).numpy()
    preds_zsl_u = logits[:, ~seen_mask].argmax(dim=1).numpy()
    guessed_zsl_u = (preds_zsl_u == test_labels_remapped_unseen)
    guessed_gzsl = (preds_gzsl == test_labels)
    zsl_unseen_acc = np.mean(
        [guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_seen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]])
    gzsl_unseen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)

    print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')
    print(f'GZSL-U: {gzsl_unseen_acc * 100:.02f}')
    print(f'GZSL-S: {gzsl_seen_acc * 100:.02f}')
    print(f'GZSL-H: {gzsl_harmonic * 100:.02f}')
    return model


def evaluate(model,args):
    run_folder = os.getcwd()
    data_folder = os.path.join(run_folder, 'data')
    train_dataset = args.train_dataset
    att_type = args.att_type
    ECN_type = args.ECN_type
    ## Similar
    print(f'<=============== Loading test data on similar ===============>')
    ds_train,ds_test,P_attrs_seen, T_attrs, T_labels = GetCrossDataset(data_folder=data_folder, train_dataset=train_dataset,test_mode='similar',att_type=att_type)
    DEVICE = 'cuda'
    num_classes = T_attrs.shape[0]
    num_seenclasses = P_attrs_seen.shape[0]
    seen_classes = sorted([i for i in range(num_seenclasses)])
    unseen_classes = sorted([i + num_seenclasses for i in range(num_classes - num_seenclasses)])
    # similar_unseen_classes = unseen_classes
    seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
    unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
    train_dataloader = DataLoader(ds_train, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=2048)

    attrs = T_attrs
    test_labels = T_labels
    class_indices_inside_test = {c: [i for i in range(len(T_labels)) if T_labels[i] == c] for c in
                                 range(num_classes)}
    test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in T_labels]
    model.eval()  # Important! Otherwise we would use unseen batch statistics
    logits = [model(x.to(DEVICE), attrs).cpu() for x, _ in test_dataloader]
    logits = torch.cat(logits, dim=0)
    # logits[:, seen_mask] *= (0.95 if train_dataset != "CUB" else 1.0)  # Trading a bit of gzsl-s for a bit of gzsl-u
    if att_type=='w2v' and train_dataset=='CUB':
        logits[:, seen_mask] *= 0.95
    else:
        logits[:, seen_mask] *= (0.95 if train_dataset != "CUB" else 1.0)  # Trading a bit of gzsl-s for a bit of gzsl-u
    preds_gzsl = logits.argmax(dim=1).numpy()
    preds_zsl_s = logits[:, seen_mask].argmax(dim=1).numpy()
    preds_zsl_u = logits[:, ~seen_mask].argmax(dim=1).numpy()
    guessed_zsl_u = (preds_zsl_u == test_labels_remapped_unseen)
    guessed_gzsl = (preds_gzsl == test_labels)
    zsl_unseen_acc = np.mean(
        [guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_seen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]])
    gzsl_unseen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)

    print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')
    print(f'GZSL-U: {gzsl_unseen_acc * 100:.02f}')
    print(f'GZSL-S: {gzsl_seen_acc * 100:.02f}')
    print(f'GZSL-H: {gzsl_harmonic * 100:.02f}')

    ## Dissimilar
    print(f'<=============== Loading test data on dissimilar ===============>')
    ds_train,ds_test,P_attrs_seen, T_attrs, T_labels = GetCrossDataset(data_folder=data_folder, train_dataset=train_dataset,test_mode='dissimilar',att_type=att_type)
    DEVICE = 'cuda'
    num_classes = T_attrs.shape[0]
    num_seenclasses = P_attrs_seen.shape[0]
    seen_classes = sorted([i for i in range(num_seenclasses)])
    unseen_classes = sorted([i + num_seenclasses for i in range(num_classes - num_seenclasses)])
    seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
    unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
    train_dataloader = DataLoader(ds_train, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=2048)

    attrs = T_attrs
    test_labels = T_labels
    class_indices_inside_test = {c: [i for i in range(len(T_labels)) if T_labels[i] == c] for c in
                                 range(num_classes)}
    test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in T_labels]
    model.eval()  # Important! Otherwise we would use unseen batch statistics
    logits = [model(x.to(DEVICE), attrs).cpu() for x, _ in test_dataloader]
    logits = torch.cat(logits, dim=0)
    logits[:, seen_mask] *= (0.95 if train_dataset != "CUB" else 1.0)  # Trading a bit of gzsl-s for a bit of gzsl-u
    preds_gzsl = logits.argmax(dim=1).numpy()
    preds_zsl_s = logits[:, seen_mask].argmax(dim=1).numpy()
    preds_zsl_u = logits[:, ~seen_mask].argmax(dim=1).numpy()
    guessed_zsl_u = (preds_zsl_u == test_labels_remapped_unseen)
    guessed_gzsl = (preds_gzsl == test_labels)
    zsl_unseen_acc = np.mean(
        [guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_seen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]])
    gzsl_unseen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)

    print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')
    print(f'GZSL-U: {gzsl_unseen_acc * 100:.02f}')
    print(f'GZSL-S: {gzsl_seen_acc * 100:.02f}')
    print(f'GZSL-H: {gzsl_harmonic * 100:.02f}')

    ## Similar & Dissimilar
    print(f'<=============== Loading test data on similar&dissimilar ===============>')
    ds_train,ds_test,P_attrs_seen, T_attrs, T_labels = GetCrossDataset(data_folder=data_folder, train_dataset=train_dataset,test_mode='similar&dissimilar',att_type=att_type)
    DEVICE = 'cuda'
    num_classes = T_attrs.shape[0]
    num_seenclasses = P_attrs_seen.shape[0]
    seen_classes = sorted([i for i in range(num_seenclasses)])
    unseen_classes = sorted([i + num_seenclasses for i in range(num_classes - num_seenclasses)])
    seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
    unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
    train_dataloader = DataLoader(ds_train, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=2048)

    attrs = T_attrs
    test_labels = T_labels
    class_indices_inside_test = {c: [i for i in range(len(T_labels)) if T_labels[i] == c] for c in
                                 range(num_classes)}
    test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in T_labels]
    model.eval()  # Important! Otherwise we would use unseen batch statistics
    logits = [model(x.to(DEVICE), attrs).cpu() for x, _ in test_dataloader]
    logits = torch.cat(logits, dim=0)
    logits[:, seen_mask] *= (0.95 if train_dataset != "CUB" else 1.0)  # Trading a bit of gzsl-s for a bit of gzsl-u
    preds_gzsl = logits.argmax(dim=1).numpy()
    preds_zsl_s = logits[:, seen_mask].argmax(dim=1).numpy()
    preds_zsl_u = logits[:, ~seen_mask].argmax(dim=1).numpy()
    guessed_zsl_u = (preds_zsl_u == test_labels_remapped_unseen)
    guessed_gzsl = (preds_gzsl == test_labels)
    zsl_unseen_acc = np.mean(
        [guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_seen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]])
    gzsl_unseen_acc = np.mean(
        [guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)

    print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')
    print(f'GZSL-U: {gzsl_unseen_acc * 100:.02f}')
    print(f'GZSL-S: {gzsl_seen_acc * 100:.02f}')
    print(f'GZSL-H: {gzsl_harmonic * 100:.02f}')
    return zsl_unseen_acc, gzsl_harmonic


if __name__ == "__main__":
    args = get_parser()
    seed = args.seed
    best_gzsl = 0
    best_zsl=0
    best_seed= 0
    metric_list = []

    model = train(args)
    zsl, gzsl = evaluate(model,args)
    if gzsl > best_gzsl:
        best_gzsl = gzsl
        best_zsl = zsl
        best_seed = seed
    print("best seed:{}|best gzsl:{}|best zsl:{}".format(best_seed, best_gzsl, best_zsl))
    metric_list.append({"seed:{}|zsl:{}|gzsl:{}".format(seed, zsl, gzsl)})
