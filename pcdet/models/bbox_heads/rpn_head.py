import torch
import torch.nn as nn
import numpy as np
from functools import partial
from ..model_utils.pytorch_utils import Empty, Sequential
from .anchor_target_assigner import AnchorGeneratorRange, TargetAssigner
from ...utils import box_coder_utils, common_utils, loss_utils
from ...config import cfg
from PIL import Image
import torch.nn.functional as F
class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_bases=9, use_dis=False, with_share=True, is_share_coef=False, return_base=False):
        super(DynamicConv2d, self).__init__()
        self.num_bases = num_bases
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.use_dis = use_dis
        self.return_base = return_base
        self.shared_conv = nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=self.kernel_size,
                                  stride=stride,
                                  padding=self.padding,
                                  dilation=dilation,
                                  groups=groups)
        conv_list = []
        for i in range(self.num_bases):
            conv = nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          kernel_size=self.kernel_size,
                          stride=stride,
                          padding=self.padding,
                          dilation=dilation,
                          groups=groups)
            conv_list.append(conv)
        self.base_convs = nn.ModuleList(conv_list)
        if use_dis:
            coef_in_channels = self.in_channels + 1
        else:
            coef_in_channels = self.in_channels
        self.base_coef = nn.Sequential(
            nn.Conv2d(in_channels=coef_in_channels,
                      out_channels=int(self.in_channels / 8),
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(self.in_channels / 8),
                      out_channels=self.num_bases,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups)
        )
        if self.use_dis:
            x_dis = np.arange(0, 248) + 0.5 - 124
            x_dis = x_dis * 0.32
            y_dis = np.arange(0, 216) + 0.5
            y_dis = y_dis * 0.32
            x_dis = np.tile(np.expand_dims(x_dis, 1), [1, 216])
            y_dis = np.tile(np.expand_dims(y_dis, 0), [248, 1])
            dis = np.sqrt(np.power(x_dis, 2) + np.power(y_dis, 2)) / 10
            self.dis = torch.tensor(dis).cuda()

    def forward(self, input):
        """
        :param input: [B,C,H,W]
        :param base_coef: [B,num_base,H,W]
        :return:
        """
        shared_output = self.shared_conv(input)
        if self.use_dis:
            bs = input.size()[0]
            stride = int(248 / input.size()[2])
            dis = self.dis.unsqueeze(0).unsqueeze(0)
            dis = F.max_pool2d(dis, kernel_size=stride, stride=stride)
            dis = dis.repeat([bs, 1, 1, 1]).type(input.dtype)
            base_coef = self.base_coef(torch.cat([input, dis], dim=1))
        else:
            base_coef = self.base_coef(input)
        if self.num_bases > 1:
            base_coef = F.softmax(base_coef, dim=1)
        else:
            base_coef = F.sigmoid(base_coef)
        output_list = []
        for i in range(self.num_bases):
            output_list.append(self.base_convs[i](input).unsqueeze(1))
        if self.num_bases>1:
            output = torch.cat(output_list, dim=1)  # [B,num_base,C_Out,H,W]
        else:
            output = output_list[0]
        output = output.permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,num_base,C_Out]
        base_coef = base_coef.permute(0, 2, 3, 1).contiguous().unsqueeze(-1)  # [B,H,W,num_base,1]
        output = output * base_coef
        output = output.sum(3)
        output = output.permute(0, 3, 1, 2).contiguous()
        output = output + shared_output
        return output

class AttFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttFusion, self).__init__()
        self.avg_pooling = self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att_conv = nn.Sequential(
            nn.Conv2d(2*in_channels,2,1),
            nn.Sigmoid())
    def forward(self,x):
        x_pool = self.avg_pooling(x)
        size = x.size()
        x_pool = x_pool.repeat(1,1,size[2],size[3])
        x = torch.cat([x,x_pool],1)
        # x = x-x_pool
        attention = self.att_conv(x)
        return attention

class SEFusion(nn.Module):
    def __init__(self, in_channels):
        super(SEFusion, self).__init__()
        reduction=4
        self.fc = nn.Sequential(
            nn.Linear(2*in_channels, 2*in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2*in_channels // reduction, 2*in_channels, bias=False),
            nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x1,x2):
        x = torch.cat([x1,x2],1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        output = x * y.expand_as(x)
        output1 = output[:,0:int(c/2),:,:]
        output2 = output[:,int(c/2):c,:,:]
        return output1,output2


class AnchorHead(nn.Module):
    def __init__(self, grid_size, anchor_target_cfg):
        super().__init__()

        anchor_cfg = anchor_target_cfg.ANCHOR_GENERATOR
        anchor_generators = []

        self.num_class = len(cfg.CLASS_NAMES)
        for cur_name in cfg.CLASS_NAMES:
            cur_cfg = None
            for a_cfg in anchor_cfg:
                if a_cfg['class_name'] == cur_name:
                    cur_cfg = a_cfg
                    break
            assert cur_cfg is not None, 'Not found anchor config: %s' % cur_name
            anchor_generator = AnchorGeneratorRange(
                anchor_ranges=cur_cfg['anchor_range'],
                sizes=cur_cfg['sizes'],
                rotations=cur_cfg['rotations'],
                class_name=cur_cfg['class_name'],
                match_threshold=cur_cfg['matched_threshold'],
                unmatch_threshold=cur_cfg['unmatched_threshold']
            )
            anchor_generators.append(anchor_generator)

        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)()

        self.target_assigner = TargetAssigner(
            anchor_generators=anchor_generators,
            pos_fraction=anchor_target_cfg.SAMPLE_POS_FRACTION,
            sample_size=anchor_target_cfg.SAMPLE_SIZE,
            region_similarity_fn_name=anchor_target_cfg.REGION_SIMILARITY_FN,
            box_coder=self.box_coder
        )
        self.num_anchors_per_location = self.target_assigner.num_anchors_per_location
        self.box_code_size = self.box_coder.code_size

        feature_map_size = grid_size[:2] // anchor_target_cfg.DOWNSAMPLED_FACTOR
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = self.target_assigner.generate_anchors(feature_map_size)
        anchors_dict = self.target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret['anchors'].reshape([-1, 7])
        self.anchor_cache = {
            'anchors': anchors,
            'anchors_dict': anchors_dict,
        }

        self.forward_ret_dict = None
        self.build_losses(cfg.MODEL.LOSSES)

    def build_losses(self, losses_cfg):
        # loss function definition
        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']

        rpn_code_weights = code_weights[3:7] if losses_cfg.RPN_REG_LOSS == 'bin-based' else code_weights
        self.reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=rpn_code_weights)
        self.dir_loss_func = loss_utils.WeightedSoftmaxClassificationLoss()

    def assign_targets(self, gt_boxes):
        """
        :param gt_boxes: (B, N, 8)
        :return:
        """
        gt_boxes = gt_boxes.cpu().numpy()
        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, 7]
        gt_boxes = gt_boxes[:, :, :7]
        targets_dict_list = []
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]

            cur_gt_classes = gt_classes[k][:cnt + 1]
            cur_gt_names = np.array(cfg.CLASS_NAMES)[cur_gt_classes.astype(np.int32) - 1]
            cur_target_dict = self.target_assigner.assign_v2(
                anchors_dict=self.anchor_cache['anchors_dict'],
                gt_boxes=cur_gt,
                gt_classes=cur_gt_classes,
                gt_names=cur_gt_names
            )
            targets_dict_list.append(cur_target_dict)

        targets_dict = {}
        for key in targets_dict_list[0].keys():
            val = np.stack([x[key] for x in targets_dict_list], axis=0)
            targets_dict[key] = val

        return targets_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim+1]) * torch.cos(boxes2[..., dim:dim+1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim+1]) * torch.sin(boxes2[..., dim:dim+1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim+1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim+1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period_torch(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_loss(self, forward_ret_dict=None):
        loss_cfgs = cfg.MODEL.LOSSES
        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict

        anchors = forward_ret_dict['anchors']
        box_preds = forward_ret_dict['box_preds']
        cls_preds = forward_ret_dict['cls_preds']
        box_dir_cls_preds = forward_ret_dict['dir_cls_preds']
        box_cls_labels = forward_ret_dict['box_cls_labels']
        box_reg_targets = forward_ret_dict['box_reg_targets']
        batch_size = int(box_preds.shape[0])

        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        # rpn head losses
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        num_class = self.num_class

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), num_class + 1, dtype=box_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        if cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros']:
            cls_preds = cls_preds.view(batch_size, -1, num_class)
            one_hot_targets = one_hot_targets[..., 1:]
        else:
            cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

        loss_weights_dict = loss_cfgs.LOSS_WEIGHTS
        cls_loss = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss_reduced = cls_loss.sum() / batch_size
        cls_loss_reduced = cls_loss_reduced * loss_weights_dict['rpn_cls_weight']

        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location)
        if loss_cfgs.RPN_REG_LOSS == 'smooth-l1':
            # sin(a - b) = sinacosb-cosasinb
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
            loc_loss = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
            loc_loss_reduced = loc_loss.sum() / batch_size
        else:
            raise NotImplementedError

        loc_loss_reduced = loc_loss_reduced * loss_weights_dict['rpn_loc_weight']

        rpn_loss = loc_loss_reduced + cls_loss_reduced

        tb_dict = {
            'rpn_loss_loc': loc_loss_reduced.item(),
            'rpn_loss_cls': cls_loss_reduced.item()
        }
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS['dir_offset'],
                num_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins']
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins'])
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * loss_weights_dict['rpn_dir_weight']
            rpn_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict


class RPNV2(AnchorHead):
    def __init__(self, num_class, args, anchor_target_cfg, grid_size, **kwargs):
        super().__init__(grid_size=grid_size, anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        if args['encode_background_as_zeros']:
            num_cls = self.num_anchors_per_location * num_class
        else:
            num_cls = self.num_anchors_per_location * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = self.num_anchors_per_location * self.box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)
        if args['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(c_in, self.num_anchors_per_location * args['num_direction_bins'], 1)

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, x_in, bev=None, **kwargs):
        ups = []
        x = x_in
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)
        ret_dict['spatial_features_last'] = x

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({
            'box_preds': box_preds,
            'cls_preds': cls_preds,
        })
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds

        ret_dict['anchors'] = torch.from_numpy(self.anchor_cache['anchors']).cuda()
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )

            ret_dict.update({
                'box_cls_labels': torch.from_numpy(targets_dict['labels']).cuda(),
                'box_reg_targets': torch.from_numpy(targets_dict['bbox_targets']).cuda(),
                'reg_src_targets': torch.from_numpy(targets_dict['bbox_src_targets']).cuda(),
                'reg_weights': torch.from_numpy(targets_dict['bbox_outside_weights']).cuda(),
            })

        self.forward_ret_dict = ret_dict
        return ret_dict

class RPNV2_Expand(AnchorHead):
    def __init__(self, num_class, args, anchor_target_cfg, grid_size, **kwargs):
        super().__init__(grid_size=grid_size, anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []
        blocks_expand = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                if j == layer_num-1:
                        block.add(DynamicConv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                else:
                    block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        for i, layer_num in enumerate(args['layer_nums'][0:-1]):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                if j == layer_num-1:
                    block.add(DynamicConv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                else:
                    block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks_expand.append(block)
        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.blocks_expand = nn.ModuleList(blocks_expand)
        self.deblocks = nn.ModuleList(deblocks)

        if args['encode_background_as_zeros']:
            num_cls = self.num_anchors_per_location * num_class
        else:
            num_cls = self.num_anchors_per_location * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = self.num_anchors_per_location * self.box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)
        if args['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(c_in, self.num_anchors_per_location * args['num_direction_bins'], 1)
        self.att_net = AttFusion(args['num_filters'][0])
        # self.se_net = SEFusion(args['num_filters'][1])
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, x_in_list,bev=None, **kwargs):
        ups = []
        x_in = x_in_list[0]
        x_in_expand = x_in_list[1]
        x = x_in
        x_expand = x_in_expand
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i<len(self.blocks)-1:
                x_expand = self.blocks_expand[i](x_expand)
                if i == 0:
                    attention = self.att_net(x_expand)+1
                    x = x.permute(0,2,3,1) * attention.permute(0,2,3,1)[:,:,:,0:1]
                    x = x.permute(0,3,1,2)
                    x_expand= x_expand.permute(0,2,3,1) * attention.permute(0,2,3,1)[:,:,:,1:2]
                    x_expand = x_expand.permute(0,3,1,2)
                x = x+x_expand
                x_expand = x
            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)
        ret_dict['spatial_features_last'] = x

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({
            'box_preds': box_preds,
            'cls_preds': cls_preds,
        })
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds

        ret_dict['anchors'] = torch.from_numpy(self.anchor_cache['anchors']).cuda()
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )

            ret_dict.update({
                'box_cls_labels': torch.from_numpy(targets_dict['labels']).cuda(),
                'box_reg_targets': torch.from_numpy(targets_dict['bbox_targets']).cuda(),
                'reg_src_targets': torch.from_numpy(targets_dict['bbox_src_targets']).cuda(),
                'reg_weights': torch.from_numpy(targets_dict['bbox_outside_weights']).cuda(),
            })

        self.forward_ret_dict = ret_dict
        return ret_dict