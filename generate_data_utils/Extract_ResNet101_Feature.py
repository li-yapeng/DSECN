import os
import copy
import argparse
import numpy as np
import scipy.io as io
# os.environ["CUDA_VISIBLE_DEVICES"]='7'
import torch
import torchvision.models as models
from torch import nn
from PIL import Image

from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms


def get_transforms(transform_complex):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if transform_complex:
        train_transform = []
        size = 224
        train_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        train_transform = transforms.Compose(train_transform)
        test_transform = []
        size = 224
        test_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose(test_transform)
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ])
    return train_transform,test_transform


def get_all_image_path(base_folder,dataset_name):
    all_image_list = []
    feature_path = os.path.join(feature_folder, dataset_name, 'res101.mat')
    feature_mat = io.loadmat(feature_path)
    image_files = feature_mat['image_files']
    image_files = np.squeeze(image_files)
    for i, image_file in enumerate(image_files):
        if dataset_name=='CUB':
            image_file = image_file[0]
            image_path = base_folder + '/' + image_file.split("MSc/CUB_200_2011/")[1]
        else:
            print("Please implement the folder sturcture of {} dataset to get the image path list.".format(dataset_name))
        all_image_list.append(image_path)
    return all_image_list


class ImageDataset(Dataset):
    def __init__(self,base_folder,dataset_name):
        """
        :param base_dir: path to CUB dataset directory
        :param split: train/val
        """
        super().__init__()
        self.img_list = get_all_image_path(base_folder,dataset_name)
        _, self.transform = get_transforms(transform_complex=False)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        impath = self.img_list[index]
        image = Image.open(impath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image,impath


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.get_resnet_backbone()

    def get_resnet_backbone(self):
        resnet = models.resnet101()
        resnet_path = "../pretrained_models/resnet101-5d3b4d8f.pth"

        # 01 - load resnet to model1
        if resnet_path != None:
            state_dict = torch.load(resnet_path)
            resnet.load_state_dict(state_dict)
            # print("resnet load state dict from {}".format(opt.resnet_path))

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # self.fine_tune(True)
        self.fine_tune(fine_tune=False)

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
            # p.requires_grad = True
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                # p.requires_grad = fine_tune
                p.requires_grad = False

    def forward(self,image):
        x = self.resnet(image)
        return x


def updata_featuremat(feature):
    feature_path = os.path.join(feature_folder, dataset_name, 'res101.mat')
    feature_mat = io.loadmat(feature_path)
    new_feature_mat = copy.deepcopy(feature_mat)
    new_feature_mat['features']= np.transpose(feature)
    return new_feature_mat


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default="")
    parser.add_argument('--feature_folder', type=str, default="")
    parser.add_argument('--save_folder', type=str, default='../generated_features')
    parser.add_argument('--dataset_name', type=str, default='CUB')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    batch_size = 1024
    args = get_parser()
    dataset_name = args.dataset_name
    save_dataset_folder = os.path.join(args.save_folder,dataset_name)
    if not os.path.exists(save_dataset_folder):
        os.makedirs(save_dataset_folder)
    save_path = os.path.join(save_dataset_folder,'res101_useweight.mat')
    print('save_path:{}'.format(save_path))
    base_folder = args.image_folder
    feature_folder = os.path.join(args.feature_folder,'xlsa17',"data")

    model = ResNetEncoder().cuda()
    model.eval()
    dataset = ImageDataset(base_folder,dataset_name)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    feature_list = []
    path_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image,path = data
            image = image.to('cuda')
            feature = model(image)
            feature = feature.squeeze(2).squeeze(2)
            feature_list.append(feature.detach().cpu())
            path_list = path_list + list(path)
            print("run {}".format(len(path_list)))
        feature = torch.cat(feature_list,dim=0).numpy()
    new_feature_mat = updata_featuremat(feature)
    io.savemat(save_path,new_feature_mat)


