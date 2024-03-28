import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torch import nn


class DSECN(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int, dropout_rate=0.3, args=None,DSE_weight=None,HTE_weight=None):
        # sun:dropout_rate=0; AWA2,CUB:dropout_rate=0.3
        super().__init__()

        self.use_data, self.use_DSE, self.use_HTE = args.use_data,args.use_DSE,args.use_HTE
        self.ECN_type = args.ECN_type
        self.att_type = args.att_type
        self.dataset = args.train_dataset

        self.S2V = nn.Sequential(

            nn.Linear(attr_dim, hid_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, 2048),
            nn.LayerNorm(proto_dim),
            nn.ReLU(),

        )

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

        print("use_data:{} | use_DSE:{} | use_HTE:{}".format(self.use_data, self.use_DSE, self.use_HTE))


        if self.att_type=='clip':
            # CUB
            if self.dataset == 'CUB':
                weight_DSE = 1
                weight_HTE = 1
            elif self.dataset == 'AWA2':
                # # AWA2
                weight_DSE = 10
                weight_HTE = 0.01
            elif self.dataset == 'SUN':
                # SUN
                weight_DSE = 100
                weight_HTE = 1
        else:
            if self.dataset == 'CUB':
                weight_DSE = 2
                weight_HTE = 10
            elif self.dataset == 'AWA2':
                # # AWA2
                weight_DSE = 1
                weight_HTE = 1
            elif self.dataset == 'SUN':
                # SUN
                weight_DSE = 2
                weight_HTE = 1
        if DSE_weight is None:
            self.weight_DSE = weight_DSE
        else:
            self.weight_DSE = DSE_weight
        if HTE_weight is None:
            self.weight_HTE = weight_HTE
        else:
            self.weight_HTE = HTE_weight
        print('weight_DSE:{}|weight_HTE:{}'.format(self.weight_DSE, self.weight_HTE))

    def train_loss(self, attrs_seen, feats, targets, clip_1k, label_1k, clip_hypo, label_hypo):
        # use_data, use_1k, use_hypo = True, True, True
        use_data, use_DSE, use_HTE = self.use_data, self.use_DSE, self.use_HTE
        # print("use_data:{} | use_1k:{} | use_hypo:{}".format(use_data,use_1k,use_hypo))
        weight_DSE, weight_HTE = self.weight_DSE,self.weight_HTE
        total_loss = 0

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

    def forward(self, x, attrs, train=False):
        if train:
            feat = self.S2V(attrs)
            logit = self.resnet_classifier(feat)
            return logit
        else:
            feat = x
            protos = self.S2V(attrs)
            x_ns = 5 * feat / feat.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
            protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True)  # [num_classes, x_dim]
            logits = x_ns @ protos_ns.t()  # [batch_size, num_classes]

            return logits



if __name__=="__main__":
    model = DSECN(512,256,2048)