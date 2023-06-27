import torch
import torch.nn as nn
import torch.nn.functional as F

class MPJPELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
    def forward(self,pred_joints,gt_joints):
        loss = torch.tensor(0).float().to(torch.device('cuda'))
        bs,point_num,_ = pred_joints.shape
        for i in range(bs):
            pred_sk = pred_joints[i]
            gt_sk = gt_joints[i]
            for j in range(1,point_num):
                loss += torch.norm(pred_sk[j] - gt_sk[j])/torch.norm(gt_sk[j])
        loss /= bs
        loss /= (point_num-1)
        return loss
class JointLengthLoss(nn.Module):
    def __init__(self,use_hand = False) -> None:
        self.use_hand = use_hand
        super().__init__()
    def _calc_length_single(self,joints):
        if not self.use_hand:
            joint_lengths = torch.zeros((20,1))
            #head
            joint_lengths[0] = torch.norm(joints[4-1]-joints[3-1])
            joint_lengths[1] = torch.norm(joints[3-1]-joints[21-1])
            #arm
            joint_lengths[2] = torch.norm(joints[21-1]-joints[9-1])
            joint_lengths[3] = torch.norm(joints[9-1]-joints[10-1])
            joint_lengths[4] = torch.norm(joints[10-1]-joints[11-1])
            joint_lengths[5] = torch.norm(joints[11-1]-joints[12-1])
            #arm
            joint_lengths[6] = torch.norm(joints[21-1]-joints[5-1])
            joint_lengths[7] = torch.norm(joints[5-1]-joints[6-1])
            joint_lengths[8] = torch.norm(joints[6-1]-joints[7-1])
            joint_lengths[9] = torch.norm(joints[7-1]-joints[8-1])
            #body
            joint_lengths[10] = torch.norm(joints[21-1]-joints[2-1])
            joint_lengths[11] = torch.norm(joints[2-1]-joints[1-1])
            #leg
            joint_lengths[12] = torch.norm(joints[1-1]-joints[17-1])
            joint_lengths[13] = torch.norm(joints[17-1]-joints[18-1])
            joint_lengths[14] = torch.norm(joints[18-1]-joints[19-1])
            joint_lengths[15] = torch.norm(joints[19-1]-joints[20-1])
            #leg
            joint_lengths[16] = torch.norm(joints[1-1]-joints[13-1])
            joint_lengths[17] = torch.norm(joints[13-1]-joints[14-1])
            joint_lengths[18] = torch.norm(joints[14-1]-joints[15-1])
            joint_lengths[19] = torch.norm(joints[15-1]-joints[16-1])

        else:
            raise
        return joint_lengths
    def forward(self,pred_joints,gt_joints):
        bs = pred_joints.shape[0]
        loss = torch.tensor(0).float().to(torch.device('cuda'))
        for i in range(bs):
            pred_sk = pred_joints[i]
            gt_sk = gt_joints[i]
            pred_length = self._calc_length_single(pred_sk)
            gt_length = self._calc_length_single(gt_sk)
            length_diff = torch.abs(pred_length - gt_length)/gt_length
            bone_num,_ = length_diff.shape
            length_diff = torch.sum(length_diff)/bone_num
            loss += length_diff
        loss /= bs
        return loss
class SymmetryLoss(nn.Module):
    def __init__(self,use_hand = False) -> None:
        self.use_hand = use_hand
        super().__init__()
    def _calc_length_single(self,joints):
        if not self.use_hand:
            joint_lengths = torch.zeros((20,1))
            #head
            joint_lengths[0] = torch.norm(joints[4-1]-joints[3-1])
            joint_lengths[1] = torch.norm(joints[3-1]-joints[21-1])
            #arm
            joint_lengths[2] = torch.norm(joints[21-1]-joints[9-1])
            joint_lengths[3] = torch.norm(joints[9-1]-joints[10-1])
            joint_lengths[4] = torch.norm(joints[10-1]-joints[11-1])
            joint_lengths[5] = torch.norm(joints[11-1]-joints[12-1])
            #arm
            joint_lengths[6] = torch.norm(joints[21-1]-joints[5-1])
            joint_lengths[7] = torch.norm(joints[5-1]-joints[6-1])
            joint_lengths[8] = torch.norm(joints[6-1]-joints[7-1])
            joint_lengths[9] = torch.norm(joints[7-1]-joints[8-1])
            #body
            joint_lengths[10] = torch.norm(joints[21-1]-joints[2-1])
            joint_lengths[11] = torch.norm(joints[2-1]-joints[1-1])
            #leg
            joint_lengths[12] = torch.norm(joints[1-1]-joints[17-1])
            joint_lengths[13] = torch.norm(joints[17-1]-joints[18-1])
            joint_lengths[14] = torch.norm(joints[18-1]-joints[19-1])
            joint_lengths[15] = torch.norm(joints[19-1]-joints[20-1])
            #leg
            joint_lengths[16] = torch.norm(joints[1-1]-joints[13-1])
            joint_lengths[17] = torch.norm(joints[13-1]-joints[14-1])
            joint_lengths[18] = torch.norm(joints[14-1]-joints[15-1])
            joint_lengths[19] = torch.norm(joints[15-1]-joints[16-1])

        else:
            raise
        return joint_lengths
    def _calc_symmetry_diff(self,length):
        side_arm_length = length[2:6]
        other_side_arm_length = length[6:10]
        side_leg_length = length[12:16]
        other_side_leg_length = length[16:]
        side = torch.concat((side_arm_length,side_leg_length))
        other_side = torch.concat((other_side_arm_length,other_side_leg_length))
        diff_r = torch.abs(side/other_side-1)

        diff_r = torch.sum(diff_r)/8
        return diff_r


    def forward(self,pred_joints):
        bs = pred_joints.shape[0]
        loss = torch.tensor(0).float().to(torch.device('cuda'))
        for i in range(bs):
            pred_sk = pred_joints[i]
            pred_length = self._calc_length_single(pred_sk)
            diff_r = self._calc_symmetry_diff(pred_length)
            loss += diff_r
        loss /= bs
        return loss