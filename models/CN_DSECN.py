import torch
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """

    def __init__(self, feat_dim: int):
        super().__init__()

        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)

    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)

            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)

            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)

        return result


class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int, USE_DSECN=False,dataset=''):
        super().__init__()
        self.USE_DSECN = USE_DSECN
        self.DEVICE = 'cuda'
        self.dataset = dataset
        USE_CLASS_STANDARTIZATION = True
        USE_PROPER_INIT = True

        if USE_DSECN:
            np.random.seed(1)
            torch.manual_seed(1)
            self.use_data, self.use_DSE, self.use_HTE = True, True, True
        else:
            np.random.seed(1)
            torch.manual_seed(1)
            self.use_data, self.use_DSE, self.use_HTE = True, False, False

        self.model = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.ReLU(),

            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.Linear(hid_dim, proto_dim),
            nn.ReLU(),
        )

        if USE_PROPER_INIT:
            weight_var = 1 / (hid_dim * proto_dim)
            b = np.sqrt(3 * weight_var)
            self.model[-2].weight.data.uniform_(-b, b)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, gamma=0.1, step_size=25)
        if USE_DSECN:
            resnet = models.resnet101()
            num_ftrs = resnet.fc.in_features
            resnet_path = "/mnt/ssd/liyapeng/research/GZSL/code/APN-ZSL/pretrained_models/resnet101-5d3b4d8f.pth"
            num_fc = 1000
            resnet.fc = nn.Linear(num_ftrs, num_fc)

            # 01 - load resnet to model1
            if resnet_path != None:
                state_dict = torch.load(resnet_path)
                resnet.load_state_dict(state_dict)
                # print("resnet load state dict from {}".format(resnet_path))

            self.resnet_classifier = list(resnet.children())[-1]
            for p in self.resnet_classifier.parameters():
                p.requires_grad = False

    def forward(self, x, attrs, train=False):
        if train:
            feat = self.model(attrs)
            logit = self.resnet_classifier(feat)
            return logit
        else:
            protos = self.model(attrs)
            x_ns = 5 * x / x.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
            protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True)  # [num_classes, x_dim]
            logits = x_ns @ protos_ns.t()  # [batch_size, num_classes]
            return logits

    def train_loss(self, attrs_seen, feats, targets, clip_1k, label_1k, clip_hypo, label_hypo):
        use_data, use_DSE, use_HTE = self.use_data, self.use_DSE, self.use_HTE

        total_loss = 0
        if self.dataset=='CUB':
            # weight_DSE = 1
            # weight_HTE = 1
            weight_DSE = 10
            weight_HTE = 0.01
        elif self.dataset == 'AWA2':
            weight_DSE = 10
            weight_HTE = 0.01
        elif self.dataset == 'SUN':
            weight_DSE = 10
            weight_HTE = 0.01

        logits = self.forward(feats, attrs_seen)

        loss = F.cross_entropy(logits, targets)
        if use_data:
            total_loss = loss + total_loss

        if use_DSE:
            # update imagenet1k
            clip_1k = clip_1k / clip_1k.norm(dim=1, keepdim=True) * np.sqrt(clip_1k.shape[1])

            logits_1k = self.forward(clip_1k, clip_1k, train=True)

            loss_1k = F.cross_entropy(logits_1k, label_1k)
            total_loss = total_loss + loss_1k * weight_DSE

        if use_HTE:
            # wordnet 1k hypoclass
            clip_hypo = clip_hypo / clip_hypo.norm(dim=1, keepdim=True) * np.sqrt(clip_hypo.shape[1])
            logits_hypo = self.forward(clip_hypo, clip_hypo, train=True)
            loss_hypo = F.cross_entropy(logits_hypo, label_hypo)
            total_loss = total_loss + loss_hypo * weight_HTE
        return total_loss

    def train_loop(self,train_dataloader,epochs,attrs_seen,clip_1k,label_1k,clip_hypo,label_hypo):
        for epoch in range(epochs):
            for i, batch in enumerate(train_dataloader):
                feats = torch.from_numpy(np.array(batch[0])).to(self.DEVICE)
                targets = torch.from_numpy(np.array(batch[1])).to(self.DEVICE)
                loss = self.train_loss(attrs_seen=attrs_seen, feats=feats, targets=targets, clip_1k=clip_1k,
                                        label_1k=label_1k, clip_hypo=clip_hypo, label_hypo=label_hypo)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            if not self.USE_DSECN:
                self.scheduler.step()
            print("Epoch:{}|Loss:{}".format(epoch,loss.item()))