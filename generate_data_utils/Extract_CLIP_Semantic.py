import os
import clip
import math
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
import json
import copy
import argparse
import scipy.io as io
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_classes_names(allclasses_names):
    classes_names = []
    class_num = len(allclasses_names)
    for i in range(class_num):
        class_name = allclasses_names[i][0][0]
        if "." in class_name:
            class_name = class_name[4:]
        try:
            class_name = class_name.replace("+"," ")
        finally:
            class_name = class_name
        try:
            class_name = class_name.replace("_"," ")
        finally:
            class_name = class_name

        classes_names.append(class_name)
    return classes_names


def extract_clip_feature_by_classes(classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    class_num = text.shape[0]
    batch_size = 100
    batch_num = math.ceil(class_num / batch_size)
    feature_list = []
    # target_list = []
    for batch_idx in range(batch_num):
        start_id = batch_idx * batch_size
        end_id = (batch_idx + 1) * batch_size
        if batch_idx < (batch_num - 1):
            batch_text = text[start_id:end_id]
        else:
            batch_text = text[start_id:]
        with torch.no_grad():
            batch_text_features = model.encode_text(batch_text).cpu().numpy()
        feature_list.append(batch_text_features)
    text_features = np.concatenate(feature_list, axis=0)
    return text_features


def get_classes_names(allclasses_names):
    classes_names = []
    class_num = len(allclasses_names)
    for i in range(class_num):
        class_name = allclasses_names[i][0][0]
        if "." in class_name:
            class_name = class_name[4:]
        try:
            class_name = class_name.replace("+"," ")
        finally:
            class_name = class_name
        try:
            class_name = class_name.replace("_"," ")
        finally:
            class_name = class_name
        # classes_names.append(class_name)
        classes_names.append(class_name)
    return classes_names


def get_dataset_clip_embedding(dataset_name):
    splits_path = os.path.join(feature_folder,dataset_name,"att_splits.mat")

    attrs_mat = io.loadmat(splits_path)
    allclasses_names = attrs_mat["allclasses_names"]
    allclasses_names_cor = get_classes_names(allclasses_names)

    clip_embedding_arr = extract_clip_feature_by_classes(allclasses_names_cor)
    new_atts_mat = copy.deepcopy(attrs_mat)


    new_atts_mat['att'] = np.transpose(clip_embedding_arr)
    # new_atts_mat['original_att'] = np.transpose(clip_embedding_arr)

    io.savemat(save_path,new_atts_mat)
    print("save {}".format(save_path))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_folder', type=str, default="")
    parser.add_argument('--save_folder', type=str, default='../generated_features')
    parser.add_argument('--dataset_name', type=str, default='CUB')
    args = parser.parse_args()
    return args



if __name__=='__main__':
    args = get_parser()
    feature_folder = os.path.join(args.feature_folder,'xlsa17',"data")
    save_dataset_folder = os.path.join(args.save_folder,args.dataset_name)
    dataset_name = args.dataset_name
    if not os.path.exists(save_dataset_folder):
        os.makedirs(save_dataset_folder)
        print("Create {} Folder".format(save_dataset_folder))
    save_path = os.path.join(save_dataset_folder,"clip_splits.mat")
    get_dataset_clip_embedding(dataset_name)