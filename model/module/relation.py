"""Relation Module Definition."""
from __future__ import absolute_import

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HumanObjectRelationModule(nn.Module):
    r"""Human-object Relation Module.

    Parameters
    ----------
    num_feat: int, default is 1024
        Dimension number used in fc layers.
    num_group : int, default is 16
        Relation group number.
        dk = num_feat / num_group.
    """
    def __init__(self, num_feat=1024, num_group=16, additional_output=False):
        super(HumanObjectRelationModule, self).__init__()
        self.num_feat = num_feat
        self.num_group = num_group
        self.dim_k = int(num_feat / num_group)
        self.additional_output = additional_output

        self.fc_ctx_gt_position = nn.LazyLinear(self.num_group)
        self.fc_gt_ctx_position = nn.LazyLinear(self.num_group)

        self.fc_gt = nn.LazyLinear(self.num_feat)
        self.fc_ctx = nn.LazyLinear(self.num_feat)

        self.gt_ctx_linear_out = nn.LazyConv2d(self.num_feat, kernel_size=1, stride=1, groups=self.num_group)
        self.ctx_gt_linear_out = nn.LazyConv2d(self.num_feat, kernel_size=1, stride=1, groups=self.num_group)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, feat, ctx_feat, pose_feat, box, ctx_box, pose_box):

        relation_ho, relation_oh = self.structure_block(feat, ctx_feat, box, ctx_box)
        relation_hp, relation_ph = self.structure_block(feat, pose_feat, box, pose_box)

        relation_h = 0.5 * relation_ho + 0.5 * relation_hp

        return relation_h, relation_oh, relation_ph

    def structure_block(self, feat, ctx_feat, box, ctx_box):
        M = feat.shape[0]
        N = ctx_feat.shape[0]

        gt_ctx_pos_embedding = self.position_embedding(box, ctx_box, feat_dim=64)  # ( M*N, dim_k)
        gt_ctx_pos_feat = self.fc_gt_ctx_position(gt_ctx_pos_embedding)  # (M*N, num_group)
        gt_ctx_pos_feat = gt_ctx_pos_feat.T  # (num_group, M*N)

        ctx_gt_pos_embedding = self.position_embedding(ctx_box, box, feat_dim=64)
        ctx_gt_pos_feat = self.fc_ctx_gt_position(ctx_gt_pos_embedding)
        ctx_gt_pos_feat = ctx_gt_pos_feat.T

        gt_data = self.fc_gt(feat)
        gt_data = gt_data.view(-1, self.num_group, self.dim_k).permute(1, 0, 2)  # (num_group, M, dim_k)

        ctx_data = self.fc_ctx(ctx_feat)
        ctx_data = ctx_data.view(-1, self.num_group, self.dim_k).permute(1, 0, 2)  # (num_group, N, dim_k)

        ## HUMAN-OBJECT
        gt_ctx = torch.bmm(gt_data, ctx_data.permute(0, 2, 1))  # (num_group, M, N)
        gt_ctx = (1.0 / math.sqrt(float(self.dim_k))) * gt_ctx
        ctx_gt = gt_ctx.permute(0, 2, 1)  # (num_group, N, M)

        ## HUMAN
        gt_ctx_pos_feat = gt_ctx_pos_feat.view_as(gt_ctx)
        gt_ctx = gt_ctx.permute(1, 0, 2)  # (M, num_group, N)
        gt_ctx_pos_feat = gt_ctx_pos_feat.permute(1, 0, 2)  # (M, num_group, N)

        weighted_gt_ctx = torch.log(torch.clamp(gt_ctx_pos_feat, min=1e-6)) + gt_ctx
        weighted_gt_ctx = F.softmax(weighted_gt_ctx, dim=2)
        weighted_gt_ctx = weighted_gt_ctx.view(-1, weighted_gt_ctx.shape[-1])  # (M * num_group, N)

        if len(ctx_feat.shape) == 1:
            ctx_feat = ctx_feat.unsqueeze(0)

        gt_output = torch.mm(weighted_gt_ctx, ctx_feat)  # (M * num_group, 1024)
        gt_output = gt_output.view(-1, self.num_group * self.num_feat, 1, 1)  # (M, num_group*1024, 1, 1)
        gt_output = self.gt_ctx_linear_out(gt_output)  # (M, 1024, 1, 1)

        ## OBJECT
        ctx_gt_pos_feat = ctx_gt_pos_feat.view_as(ctx_gt)  # (num_group, N, M)
        ctx_gt = ctx_gt.permute(1, 0, 2)  # (N, num_group, M)
        ctx_gt_pos_feat = ctx_gt_pos_feat.permute(1, 0, 2)  # (N, num_group, M)

        weighted_ctx_gt = torch.log(torch.clamp(ctx_gt_pos_feat, min=1e-6)) + ctx_gt
        weighted_ctx_gt = F.softmax(weighted_ctx_gt, dim=2)
        weighted_ctx_gt = weighted_ctx_gt.view(-1, weighted_ctx_gt.shape[-1])  # (N * num_group, M)

        ctx_output = torch.mm(weighted_ctx_gt, feat.unsqueeze(0))  # (N * num_group, 1024)
        ctx_output = ctx_output.view(-1, self.num_group * self.num_feat, 1, 1)  # (N, num_group*1024, 1, 1)
        ctx_output = self.ctx_gt_linear_out(ctx_output)  # (N, 1024, 1, 1)

        # if self.additional_output:
        #     # (M * num_group, N) -> # (M, num_group, N) -> # (M, N)
        #     gt_ctx_relation = torch.mean(weighted_gt_ctx.view(M, -1, N), dim=1)
        #     return gt_output, ctx_output, gt_ctx_relation
        # else:

        return gt_output, ctx_output

    def position_embedding(self, box, ctx_box, feat_dim=64, wave_length=1000):

        # (M, 1)
        xmin, ymin, xmax, ymax = box.split(1, dim=1)
        box_width = xmax - xmin + 1.
        box_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        # (N, 1)
        ctx_xmin, ctx_ymin, ctx_xmax, ctx_ymax = ctx_box.split(1, dim=1)
        ctx_box_width = ctx_xmax - ctx_xmin + 1.
        ctx_box_height = ctx_ymax - ctx_ymin + 1.
        ctx_center_x = 0.5 * (ctx_xmin + ctx_xmax)
        ctx_center_y = 0.5 * (ctx_ymin + ctx_ymax)

        # (M, N)
        delta_x = center_x - ctx_center_x.T
        delta_x = delta_x / box_width
        delta_x = torch.log(torch.max(torch.abs(delta_x), torch.tensor(1e-4)))

        delta_y = center_y - ctx_center_y.T
        delta_y = delta_y / box_width
        delta_y = torch.log(torch.max(torch.abs(delta_y), torch.tensor(1e-4)))

        delta_width = ctx_box_width.T / box_width

        delta_width = torch.log(delta_width)

        delta_height = ctx_box_height.T / box_height
        delta_height = torch.log(delta_height)
        # # (M, N, 4)
        position_mat = torch.stack((delta_x, delta_y, delta_width, delta_height), dim=2)

        # # position embedding
        feat_range = torch.arange(0, feat_dim / 8)
        dim_mat = torch.pow(torch.full((1,), wave_length), (8. / feat_dim) * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1)  # (1, 1, 1, feat_dim/8)

        # # position_mat (M, N, 4, 1)
        position_mat = torch.unsqueeze(100.0 * position_mat, dim=3)
        div_mat = position_mat / dim_mat.to(self.device)  # (M, N, 4, feat_dim/8)

        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        embedding = torch.cat((sin_mat, cos_mat), dim=3)  # (M, N, 4, feat_dim/4)
        return embedding.view(-1, feat_dim)
