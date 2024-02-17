"""Human-object Relation Network."""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torchvision

from model.module import HumanObjectRelationModule
from lib.stanford40_dataset import Stanford40Action


class HORelationNet(nn.Module):
    def __init__(self, pretrained, features, top_features, classes,
                 transform_layer, gap_layer, roi_align,
                 roi_size, stride):

        super(HORelationNet, self).__init__()
        self.pretrained = pretrained
        self.features = features
        self.top_features = top_features
        self.classes = classes
        self.transform = transform_layer
        self.num_class = len(classes)

        self.global_avg_pool = gap_layer
        self.roi_align = roi_align
        self.roi_size = roi_size
        self.stride= stride

        self._max_batch = 1  # currently only support batch size = 1
        self.fc = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU()
        )
        self.fc_ctx = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU()
        )
        self.fc_pose = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU()
        )
        self.class_predictor = nn.LazyLinear(self.num_class)
        self.ctx_class_predictor = nn.LazyLinear(self.num_class)
        self.pose_class_predictor = nn.LazyLinear(self.num_class)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.relation = HumanObjectRelationModule(num_feat=1024, num_group=16, additional_output=False)

    def forward(self, x, h_boxes, obj_boxes, pose_boxes):

        # Nokte ****** Chon ye Bach midam pas --->
        obj_boxes = obj_boxes[0]
        pose_boxes = pose_boxes[0]

        intermediate_layer = self.features(x)

        # ROI Align for getting best Related features (14, 14)
        aligned_features_h = self.roi_align(intermediate_layer, [h_boxes])
        aligned_features_o = self.roi_align(intermediate_layer, [obj_boxes])
        aligned_features_p = self.roi_align(intermediate_layer, [pose_boxes])

        # (M, 1024, h, w) -> (M, 2048, h, w)
        top_features_h = self.top_features(aligned_features_h)
        top_features_o = self.top_features(aligned_features_o)
        top_features_p = self.top_features(aligned_features_p)

        # (M, 2048, h, w) -> (M, 2048)
        top_features_h = self.global_avg_pool(top_features_h).squeeze()
        top_features_o = self.global_avg_pool(top_features_o).squeeze()
        top_features_p = self.global_avg_pool(top_features_p).squeeze()

        # (M, 1024)
        top_feat_h = self.fc(top_features_h)
        top_feat_o = self.fc_ctx(top_features_o)
        top_feat_p = self.fc_pose(top_features_p)

        # # Relation Module
        # relation_feat, relation_ctx_feat, relation_pose_feat = \
        #     self.relation(top_feat_h, top_feat_o, top_feat_p, h_boxes, obj_boxes, pose_boxes)
        #
        # # Add Norm
        # top_feat_h = top_feat_h + relation_feat.squeeze()
        # top_feat_o = top_feat_o + relation_ctx_feat.squeeze()
        # top_feat_p = top_feat_p + relation_pose_feat.squeeze()

        # Predict Class (B * N, C) -> (B, N, C)
        cls_pred = self.class_predictor(top_feat_h)
        ctx_cls_pred = self.ctx_class_predictor(top_feat_o)
        #pose_cls_pred = self.pose_class_predictor(top_feat_p)

        ctx_cls_pred = ctx_cls_pred.max(dim=0, keepdim=True)[0].squeeze()
        #pose_cls_pred = pose_cls_pred.max(dim=0, keepdim=True)[0].squeeze()

        cls_pred = cls_pred + ctx_cls_pred  # + pose_cls_pred

        return cls_pred


def horelation_resnet50_v1d_st40(pretrained=False, transfer=None, params='', **kwargs):

    if transfer is None:
        classes = Stanford40Action.CLASSES
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Extract the backbone network (ResNet50)
        backbone = model.backbone.body

        # Extract the layers you need up to 'layer3'
        layers_to_extract = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']

        # Create a custom backbone containing only the required layers
        my_backbone = nn.Sequential()
        for name, module in backbone.named_children():
            if name in layers_to_extract:
                my_backbone.add_module(name, module)

        transform_layer = model.transform
        top_features = model.backbone.body.layer4

        w, h = 14, 14
        roi_align = torchvision.ops.RoIAlign(sampling_ratio=-1, output_size=(w, h), spatial_scale=1.0)
        gap_layer = nn.AdaptiveAvgPool2d((1, 1))

        return HORelationNet(
            pretrained=pretrained, features=my_backbone, top_features=top_features,
            classes=classes,transform_layer=transform_layer, gap_layer=gap_layer,
            roi_align=roi_align, roi_size=(14, 14), stride=16)
    else:
        raise NotImplementedError
