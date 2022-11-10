# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from random import randint
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
import math
import kornia as K

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
        
def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

class DeformableTransformer(nn.Module):
    def __init__(self, device, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.device = torch.device(device)
        self.two_stage_num_proposals = two_stage_num_proposals
        
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        # used in method get_query_embedd_from_pred_atten_map to adjust the concentration ratio of sampling  
        self.sampling_ratio = nn.Parameter(torch.tensor(1.0))
        # used in combine tgt from points and tgt from embedding
        self.tgt_ratio = nn.Embedding(300,2)

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)
            self.pos_trans = MLP(d_model, d_model*2,  d_model*2, 2)
            self.pos_trans_norm = nn.LayerNorm(d_model*2)
            
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)
        uniform_(self.tgt_ratio.weight)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def get_global_attention(self, att_global, spatial_shapes, level_start_index):
        # (att_h,att_w)=spatial_shapes[0]
        # output = att_global[:,:,:level_start_index[1]].view(-1,att_h,att_w)
        # return output

        bs = att_global.shape[0]
        att_list = []
        for i, (H_, W_) in enumerate(spatial_shapes):
            if i == 0:
                att_h, att_w = H_, W_
                att_list.append(att_global[:,:,:level_start_index[1]].view(-1,att_h,att_w))
                continue
            if i == len(level_start_index)-1:
                att = att_global[:,:,level_start_index[i]:]
            else:
                att = att_global[:,:,level_start_index[i]:level_start_index[i+1]]
            n_h = math.ceil(att_h/float(H_))
            m = nn.Upsample(scale_factor=n_h, mode='bilinear')
            att = m(att.view(bs,1,H_, W_)).squeeze(1)
            if att.shape[-1] > att_w or att.shape[-2] > att_h:
                att = att[:,:att_h,:att_w]
            att_list.append(att)
        output = att_list[0] + att_list[1] + att_list[2] + att_list[3]
        return output
    
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, support_feature, orignal_shape, num_queries, true_atten, mode='training',query_embed=None):
        assert self.two_stage or query_embed is not None

        #prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        dict_4_atten = {'srcs':[],'shapes':[]}
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            dict_4_atten['srcs'].append(src)
            dict_4_atten['shapes'].append([bs, c, h, w])
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        # memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)  
        memory, attens = self.encoder(src_flatten, support_feature, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)  
        # memory, attens = self.encoder(src_flatten, self.query_global.weight.unsqueeze(0), spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)  
        attmap = self.get_global_attention(attens[-1],spatial_shapes,level_start_index)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            mem = memory
            # attention maps from different encoder layer in different scale
            atten_multi_scale_aux = []
            for atten in attens:
                atten_multi_scale = []
                start_idx = 0
                for idx, (src_flatten,src_shape) in enumerate(zip(dict_4_atten['srcs'],dict_4_atten['shapes'])):
                    support_atten = atten[:,:,start_idx:start_idx+src_shape[-1]*src_shape[-2]]
                    support_atten = self.get_query_embedd_from_support(support_atten,src_shape,orignal_shape,num_queries,mode)
                    atten_multi_scale.append(support_atten)
                    start_idx = start_idx + src_shape[-1]*src_shape[-2]
                atten_multi_scale_aux.append(torch.stack(atten_multi_scale,dim=0))
            atten_multi_scale = atten_multi_scale_aux[-1]
            
            # this is implementation of original deformable detr. we use its tgt 
            _, tgt = torch.split(query_embed, c, dim=1)
            # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # reference_points = self.reference_points(query_embed).sigmoid()
            
            # uncomment code below to get reference points from groundtruth attention map
            # reference_points_v = self.get_query_embedd_from_pred_atten_map(true_atten.decompose()[0],orignal_shape,num_queries,self.sampling_ratio,mode)
            
            # get reference points from predicted attention map
            t = torch.sum(atten_multi_scale,dim=0,keepdim=False)
            reference_points_v = self.get_query_embedd_from_pred_atten_map(t/torch.max(t), orignal_shape,num_queries,self.sampling_ratio,mode)
            # reference_points_v = self.get_query_embedd_from_pred_atten_map(t/torch.max(t), orignal_shape,num_queries,self.sampling_ratio,mode)

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.pos2posemb2d(inverse_sigmoid(reference_points_v))))
            query_embed, tgt_from_p = torch.split(pos_trans_out, c, dim=2)
            
            self.tgt_ratio.weight = torch.nn.Parameter(F.softmax(self.tgt_ratio.weight,dim=-1))
            # try combine tgt from referencepoint and tgt from embedding by learnable parameter or just average
            tgt = torch.mul(self.tgt_ratio.weight[:,0].unsqueeze(-1).repeat(1,self.d_model),tgt) + torch.mul(self.tgt_ratio.weight[:,1].unsqueeze(-1).repeat(1,self.d_model),tgt_from_p)
            # tgt = 0.5*(tgt + tgt_from_p)

        init_reference_out = reference_points_v

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points_v, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        # if self.two_stage:
        #     return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        # return hs, torch.stack(atten_multi_scale,dim=0), init_reference_out, inter_references_out, None, None
        # return hs, torch.zeros_like(true_atten.decompose()[0],device=hs.device).unsqueeze(0), init_reference_out, inter_references_out, None, None,reference_points_v
        return hs,torch.stack(atten_multi_scale_aux,dim=0), attmap, init_reference_out, inter_references_out, None, None,reference_points_v

    def get_query_embedd_from_support(self,support_atten,src_shape,orignal_shape,num_queries, mode='training'):

        pred_atten_map = support_atten.reshape(src_shape[0],1,src_shape[2],src_shape[3])
        
        # upsample by keeping ratio of height and width
        ratio = max(orignal_shape[2]/float(src_shape[2]),orignal_shape[3]/float(src_shape[3]))
        n_h = math.ceil(ratio)
        m = nn.Upsample(scale_factor=n_h, mode='bilinear')
        pred_atten_map = m(pred_atten_map)
        if pred_atten_map.shape[-1] > orignal_shape[3] or pred_atten_map.shape[-2] > orignal_shape[2]:
            pred_atten_map = pred_atten_map[:,:,:orignal_shape[2],:orignal_shape[3]]

        # direct upsample
        # pred_atten_map = F.interpolate(pred_atten_map,size=[orignal_shape[2],orignal_shape[3]],mode='bilinear')

        return pred_atten_map

    def get_query_embedd_from_pred_atten_map(self,atten_map,orignal_shape,num_queries,ratio,mode='training'):
        b,c,h,w = orignal_shape
        # atten_map = K.filters.laplacian(atten_map, kernel_size=5).clamp(0., 1.)
        atten_map_flatten = atten_map.flatten(1)
        # atten_map_flatten = F.softmax(atten_map_flatten/ratio,dim = -1)
        atten_map_flatten = F.softmax(atten_map_flatten/0.1, dim = -1)
        query_index = torch.multinomial(atten_map_flatten, num_queries, replacement=True)

        query_postions = []
        for i in range(b):
            query_postion = self.index2pos(query_index[i],h,w)
            query_postions.append(query_postion)
        
        return torch.stack(query_postions,dim=0)
    
    def index2pos(self,index,h,w):
        num_queries = 0
        if index.dim()>1:
            num_queries = index.shape[1]
            index = index.squeeze()
        else:
            num_queries = index.shape[0]

        pos = torch.zeros(num_queries,2,device=self.device)

        for i in range(num_queries):
            ix = (index[i] // w) /float(h)
            iy = (index[i] % w) / float(w)

            pos[i][0] = iy
            pos[i][1] = ix

        return pos
        
    def pos2posemb2d(self, pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        #pos = pos * scale
        pos[..., 0] = pos[..., 0] * scale
        pos[..., 1] = pos[..., 1] * scale
        
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=self.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    def pos2posemb1d(self, pos, num_pos_feats=256, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=self.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., None] / dim_t
        posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return posemb

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.self_attn_global = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, support, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # 1st step: do cross attention between support feature and memory
        tgt1,atten = self.self_attn_global(support,self.with_pos_embed(src, pos).transpose(0, 1),src.transpose(0, 1))
        # 2nd step: do self attention of memory to encode information
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        # 3rd step: cat them together and add atttention results to original feature
        src = torch.cat((src,support),dim=1) + torch.cat((self.dropout1(src2),tgt1),1)
        # src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src[:,:-1,:],src[:,-1:,:], atten


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, support, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # collect attention output in each layer to compute auxiliary loss
        attens = []
        for _, layer in enumerate(self.layers):
            # output is the result of memory self attention, tgt1 is the result of cross attention between memory and support feature
            # atten is the weight matrix of cross attention between memory and support feature, 
            # we use tgt1 as new support feature for next layer
            output,tgt1,atten = layer(output, support, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            attens.append(atten)
            support = tgt1
            
        return output, attens


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, d_model=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


    
def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        device=args.device)

# if __name__=='__main__':
#     DT = DeformableTransformer()
#     print(DT)



