import torch
import numpy as np
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from cc3d import connected_components

Blob_Scheduler = 0 #200000


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

# currently only for single class
class Blob_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Blob_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.calls = 0
 
    # main loss logic called on each instance separately
    def loss(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        self.calls += 1
        global_loss = self.loss(net_output, target)
        
        if self.calls < Blob_Scheduler:
            return global_loss

        if self.calls == Blob_Scheduler:
            print("Blob Loss integrated")

        blob_loss = 0.0
        num_components = 0

        for i in range(target.size(0)):
            binary_output = (torch.sigmoid(net_output[i][0]) >= 0.5).type(torch.uint8)
            target_ccs = connected_components(target[i][0]).type(torch.uint8)
            output_ccs = connected_components(binary_output).type(torch.uint8)

            ccs = [cc for cc in target_ccs.unique() if cc != 0]

            if len(ccs) > 1:

                tracked_tumors = target_ccs.max()

                cc_map = {}

                for cc in output_ccs.unique():
                    if cc != 0:
                        cc_mask = (output_ccs == cc).type(torch.uint8)

                        tumor_id = (cc_mask * target_ccs).max()

                        if tumor_id == 0:
                            tracked_tumors += 1
                            cc_map[cc] = tracked_tumors

                        else:
                            cc_map[cc] = tumor_id

                vector_map = np.vectorize(lambda x: cc_map.get(x, x))
                output_ccs = torch.tensor(vector_map(output_ccs)).type(torch.uint8) 

                for cc in ccs: 
                    total_mask = torch.ones(size=target_ccs.size()).to(target.device)

                    for j in ccs:
                        if j != cc:
                            total_mask[(target_ccs == j) | (output_ccs == j)] = 0

                    masked_output = net_output[i].unsqueeze(0) * total_mask
                    masked_target = target[i].unsqueeze(0) * total_mask

                    blob_loss += self.loss(masked_output, masked_target)
                    num_components += 1
            else:
                blob_loss += self.loss(net_output[i].unsqueeze(0), target[i].unsqueeze(0))
                num_components += 1

        blob_loss /= max(num_components, 1)
    

        return (0.3 * global_loss) + (0.7 * blob_loss)


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class Blob_DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Blob_DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        self.calls = 0

    def loss(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        self.calls += 1
        global_loss = self.loss(net_output, target)
        
        if self.calls < Blob_Scheduler:
            return global_loss

        if self.calls == Blob_Scheduler:
            print("Blob Loss integrated")

        blob_loss = 0.0
        num_components = 0

        for i in range(target.size(0)):
            binary_output = (torch.sigmoid(net_output[i][0]) >= 0.5).type(torch.uint8)
            target_ccs = connected_components(target[i][0]).type(torch.uint8)
            output_ccs = connected_components(binary_output).type(torch.uint8)

            ccs = [cc for cc in target_ccs.unique() if cc != 0]

            if len(ccs) > 1:

                tracked_tumors = target_ccs.max()

                cc_map = {}

                for cc in output_ccs.unique():
                    if cc != 0:
                        cc_mask = (output_ccs == cc).type(torch.uint8)

                        tumor_id = (cc_mask * target_ccs).max()

                        if tumor_id == 0:
                            tracked_tumors += 1
                            cc_map[cc] = tracked_tumors

                        else:
                            cc_map[cc] = tumor_id

                vector_map = np.vectorize(lambda x: cc_map.get(x, x))
                output_ccs = torch.tensor(vector_map(output_ccs)).type(torch.uint8) 

                for cc in ccs: 
                    total_mask = torch.ones(size=target_ccs.size()).to(target.device)

                    for j in ccs:
                        if j != cc:
                            total_mask[(target_ccs == j) | (output_ccs == j)] = 0

                    masked_output = net_output[i].unsqueeze(0) * total_mask
                    masked_target = target[i].unsqueeze(0) * total_mask

                    blob_loss += self.loss(masked_output, masked_target)
                    num_components += 1
            else:
                blob_loss += self.loss(net_output[i].unsqueeze(0), target[i].unsqueeze(0))
                num_components += 1

        blob_loss /= max(num_components, 1)
    

        return (0.3 * global_loss) + (0.7 * blob_loss)


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
