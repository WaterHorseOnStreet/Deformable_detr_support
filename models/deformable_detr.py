# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import device, nn, zeros_like
import math
import torchvision.transforms as transforms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from .projector import build_support_projector
import numpy as np
import torchvision
from PIL import Image,ImageDraw, ImageFont
import os
from scipy.ndimage import gaussian_filter
import kornia as K


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, projector, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.projector = projector
        self.backbone = backbone[0]
        self.backbone_support = backbone[1]
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.input_proj = nn.Conv2d(backbone[0].num_channels[-1], hidden_dim, kernel_size=1)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            print(self.backbone.num_channels)
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                print(in_channels)
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        #self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            print('----------using bbox refine-----------')
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.idx = 0

        for p in self.backbone_support.parameters():
            p.requires_grad=False
        # for p in self.backbone.parameters():
        #     p.requires_grad=False
        # for p in self.projector.parameters():
        #     p.requires_grad=False
        # for p in self.query_embed.parameters():
        #     p.requires_grad = False
        
    def get_ref_points(self, samples: NestedTensor,support_imgs, true_atten,mode='training'):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        if isinstance(support_imgs, (list, torch.Tensor)):
            support_imgs = nested_tensor_from_tensor_list(support_imgs)

        orignal_shape = samples.decompose()[0].shape

        features, pos = self.backbone(samples)

        support_feat,_ = self.backbone_support(support_imgs)


        support_embedd = self.projector(support_feat)
        # support_embedd = None
    
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        

        hs, pred_atten, pred_atten2, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, ref_points_v = self.transformer(srcs, masks, pos,support_embedd,orignal_shape,self.num_queries,true_atten,mode,query_embeds)   
        
        # if you want to plot reference points on image, uncomment code below

        return plot_ref_points(samples.decompose()[0],true_atten,ref_points_v,orignal_shape)

            

    def forward(self, samples: NestedTensor,support_imgs, true_atten,mode='training',generate_atten=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        if isinstance(support_imgs, (list, torch.Tensor)):
            support_imgs = nested_tensor_from_tensor_list(support_imgs)

        orignal_shape = samples.decompose()[0].shape

        features, pos = self.backbone(samples)

        support_feat,_ = self.backbone_support(support_imgs)


        support_embedd = self.projector(support_feat)
        # support_embedd = None
    
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        

        hs, pred_atten, pred_atten2, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, ref_points_v = self.transformer(srcs, masks, pos,support_embedd,orignal_shape,self.num_queries,true_atten,mode,query_embeds)   
        
        if generate_atten:
            pred = torch.sum(pred_atten[-1],dim=0,keepdim=False)
            pred = pred/torch.max(pred)
            ref_img = plot_ref_points(samples.decompose()[0],pred,ref_points_v,orignal_shape)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)


        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],'pred_atten': pred_atten,'pred_atten2':pred_atten2}
        if generate_atten:
            out['ref_plot'] = ref_img
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        m = 0.0
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_support.parameters()):
             param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, attens, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes,attens):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes,attens):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_attnes_focal(self, outputs, targets, indices, num_boxes, attens):
        """implement focal loss to learn the attention map
        """
        assert 'pred_atten' in outputs
        pred_attens = outputs['pred_atten']
        
        gt = attens.decompose()[0]
        pos_inds = gt.gt(0).float()
        neg_inds = gt.eq(0).float()

        neg_weights = torch.pow(1 - gt, 4)
        loss_focal = 0 
        
        for i in range(pred_attens.shape[0]):
            loss = 0
            pred_atten = pred_attens[i]
            pred_atten = torch.sum(pred_atten,dim=0,keepdim=False)
            # pred = pred_atten/torch.max(pred_atten) + 1e-6
            b,c,h,w = pred_atten.shape
            pred = pred_atten.flatten(2)
            pred = F.softmax(pred, dim=2)
            pred = pred.view(b,c,h,w)
            # pred = pred_atten.sigmoid()
            
            pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
            neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

            num_pos  = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if num_pos == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
            loss_focal = loss_focal + loss

        losses = {}

        # losses['loss_attens'] = l_kls/pred_attens.shape[0]
        losses['loss_attens'] = loss_focal/pred_attens.shape[0]
        return losses

    # loss attention is the implementation of Mingjun Li, need groundtruth attention map as input
    def loss_attens(self, outputs, targets, indices, num_boxes, attens):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        
        assert 'pred_atten' in outputs
        T = 0.1
        pred_attens = outputs['pred_atten']

        teacher_scores = attens.decompose()[0]
        l_kls = 0 
        for i in range(pred_attens.shape[0]):
            pred_atten = pred_attens[i]
            pred_atten = torch.sum(pred_atten,dim=0,keepdim=False)
            # for i in range(pred_attens.shape[0]):
            #     pred_atten = pred_attens[i]
            b,c,h,w = pred_atten.shape
            y = pred_atten.flatten(2)
            teacher_scores = teacher_scores.flatten(2)
            p = F.log_softmax(y/T, dim=2)
            q = F.log_softmax(teacher_scores/T, dim=2)
            p = p.reshape(b,c,h,w)
            q = q.reshape(b,c,h,w)
            l_kl = F.kl_div(p, q, size_average=False,log_target=True,reduction = 'batchmean') * (T**2) / b
            l_kls += l_kl

            # if i == pred_attens.shape[0] - 1:
            #     img_pred = torchvision.transforms.ToPILImage()(pred_atten[0,:,:,:])
            #     img_gt = torchvision.transforms.ToPILImage()(attens.decompose()[0][0,:,:,:])
            #     img_pred.save('pred_{}.png'.format(i))
            #     img_gt.save('gt_{}.png'.format(i))

        losses = {}

        # losses['loss_attens'] = l_kls/pred_attens.shape[0]
        losses['loss_attens'] = l_kls/pred_attens.shape[0]
        return losses
    
    # loss attention below is the implementation of Yinxian Li, generate attention map during training
    def loss_attention(self,outputs,targets, indices, num_boxes, attens):
        att = outputs['pred_atten2']
        device = att.device
        bs,att_h,att_w = att.shape
        target_boxes_list = [t['boxes'] for t in targets]
        targets_att_list = []
        for boxes in target_boxes_list:
            boxes = box_ops.box_cxcywh_to_xywh(boxes)
            target_att_one_image = torch.zeros((att_h, att_w),device=device)
            for box in boxes:
                box_x, box_y, box_w, box_h = int(box[0]*att_w), int(box[1]*att_h), int(box[2]*att_w), int(box[3]*att_h)
                if box_w == 0:
                    box_w = 1
                if box_h == 0:
                    box_h = 1  
                if box_x < 0:
                    box_x = 0
                if box_y < 0:
                    box_y = 0  
                if box_x + box_w > att_w or box_y + box_h > att_h:
                    continue
                a = torch.zeros((box_h,box_w),device=device)
                a[int(box_h/2)][int(box_w/2)] = 10
                b = torch.FloatTensor(gaussian_filter(a.cpu(), sigma=10)).cuda()

                # cc = target_att_one_image[box_y:box_y+box_h,box_x:box_x+box_w]
                # print(b.shape,cc.shape,box_h,box_w,box_x,box[0],att_w,box_y,target_att_one_image.shape)

                target_att_one_image[box_y:box_y+box_h,box_x:box_x+box_w] = target_att_one_image[box_y:box_y+box_h,box_x:box_x+box_w] + b
            targets_att_list.append(target_att_one_image.view(-1))        
        targets_att = torch.stack(targets_att_list)
        # loss_attention = F.cross_entropy(att.transpose(1, 2), targets_att)
        loss_attention = F.kl_div(att.view(bs,-1).log(), targets_att, size_average=True)
        losses = {'loss_attens': loss_attention}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_boxes,attens):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes,attens, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'attens': self.loss_attens
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        #print(outputs.keys())
        return loss_map[loss](outputs, targets, indices, num_boxes, attens, **kwargs)

    def forward(self, outputs, targets, attens):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'pred_atten'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, attens))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if loss == 'attens':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, attens, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                if loss == 'attens':
                    continue
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, attens **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class PostProcess_caltech(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        # prob = out_logits.sigmoid()
        # out_logits = out_logits[:,:200,:]
        prob = F.softmax(out_logits, -1)
        topk_values, topk_indexes = torch.topk(prob[...,1].reshape(out_logits.shape[0], -1), 300, dim=1)
        # print(topk_values)
        # print(topk_indexes)
        scores = topk_values
        topk_boxes = topk_indexes #// (out_logits.shape[2]-1)
        labels = torch.ones_like(topk_indexes,device=scores.device)#1 + (topk_indexes % (out_logits.shape[2]-1))#% (out_logits.shape[2]-1)),torch.zeros_like(topk_indexes) 
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h ,img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        for result in results:
            class_idx = torch.arange(0,300,device=scores.device)
            idx = self.non_max_suppression(class_idx, result['labels'], result['boxes'], result['scores'], 0.7)
            result['labels'] = result['labels'][idx]
            result['boxes'] = result['boxes'][idx]
            result['scores'] = result['scores'][idx]
        return results
    
    def non_max_suppression(self, pre_class_idx, pre_class, boxes, scores, threshold):
        if boxes.shape[0] == 0:
            return pre_class_idx
        # if boxes.dtype.kind != "f":
        #     boxes = boxes.astype(np.float32)
        #ixs = scores.argsort()
        ixs = pre_class_idx
        device = pre_class.device
        pick = []
        while len(ixs) > 0:
            i = ixs[0]
            pick.append(i)

            iou, _ = box_ops.box_iou(
                boxes[i].repeat(len(ixs) - 1, 1),
                boxes[ixs[1:]])
            iou = torch.diag(iou)
            remove_ixs = torch.where(iou > threshold)[0] + 1
            mask = remove_ixs.lt(0)

            for j, num in enumerate(remove_ixs):
                if pre_class[ixs[num]] == pre_class[i]:
                    mask[j] = True
            remove_ixs = remove_ixs[mask]
            ixs = np.delete(ixs.cpu(), remove_ixs.cpu())
            ixs = np.delete(ixs, 0)
        return torch.LongTensor(pick)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    
def plot_ref_points(img,pred_atten, points, origin_shape):
    transform = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # print(img.shape)
    img = transform(img)
    img = img[0,:,:,:]
    # pred = torch.sum(pred_atten[-1],dim=0,keepdim=False)
    
    # atten_map = K.filters.laplacian(pred, kernel_size=9)
    # pred = atten_map.clamp(0., 1.)
    
    pred = pred_atten[0]
    
    img[0,:,:] = pred[0] #/torch.max(pred[0])
    gt_img = transforms.ToPILImage()(img).convert('RGB')
    
    draw_gt = ImageDraw.Draw(gt_img)
    r = 5
    points = points[0,:,:]
    for p in range(points.shape[0]):
        ix = points[p][0] * origin_shape[3] 
        iy = points[p][1] * origin_shape[2]
        draw_gt.ellipse((ix-r, iy-r, ix+r, iy+r), fill=(0,255,0,0))
        
    return gt_img

#     os.makedirs(os.path.join(os.getcwd(),'./save_bbox/'), exist_ok=True)
#     save_path = os.path.join(os.getcwd(),'./save_bbox/{}.png'.format(id))
#     gt_img.save(save_path)
    
    
class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor =  transforms.functional.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)

def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    if args.dataset_file == 'cityperson' or args.dataset_file == 'caltech':
        num_classes = 91
    device = torch.device(args.device)
    print(num_classes)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    projector = build_support_projector(args)
    model = DeformableDETR(
        backbone,
        transformer,
        projector,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_attens': 1}
    # weight_dict = {'loss_ce': 0, 'loss_bbox': 0, 'loss_attens': 1}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # weight_dict['loss_giou'] = 0
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # losses = ['labels', 'boxes', 'cardinality', 'attens']
    # losses = ['attens']
    
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    if args.dataset_file == 'coco':
        postprocessors = {'bbox': PostProcess()}
    elif args.dataset_file == 'cityperson' or args.dataset_file == 'caltech':
        postprocessors = {'bbox': PostProcess_caltech()}
    else:
        postprocessors = {'bbox': PostProcess()}
        
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
