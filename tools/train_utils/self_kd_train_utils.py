'''
    This file used to test self-knowledge dsitillation PSKD
    Now In SelfEMA's get_soft_loss function may have some problem...
        1. we do not know whether the last_cls and now_cls need use the softmax to activate
        2. we do not know whether we need to use torch.nn.LogSoftmax to activate the last_cls...
        3. whether we need to normalize the cls loss....
        4. last but most important: this version SelfEMA can not used to distillation those model which use centerhead as dense head....
'''

import glob
import os
import copy

import torch
import torch.nn as nn
import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import loss_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def get_iou_consistency_loss(teacher_boxes, student_boxes):
    box_losses, cls_losses = [], []
    batch_normalizer = 0
    for teacher_box, student_box in zip(teacher_boxes, student_boxes):
        teacher_cls_preds = teacher_box['pred_cls_preds'].detach_()
        teacher_box_preds = teacher_box['pred_boxes'].detach_()
        student_cls_preds = student_box['pred_cls_preds']
        student_box_preds = student_box['pred_boxes']
        num_teacher_boxes = teacher_box_preds.shape[0]
        num_student_boxes = student_box_preds.shape[0]
        if num_teacher_boxes == 0 or num_student_boxes == 0:
            batch_normalizer += 1
            continue

        with torch.no_grad():
            teacher_class = torch.max(teacher_cls_preds, dim=-1, keepdim=True)[1] # [Nt, 1]
            student_class = torch.max(student_cls_preds, dim=-1, keepdim=True)[1] # [Ns, 1]
            not_same_class = (teacher_class != student_class.T).float() # [Nt, Ns]

            iou_3d = boxes_iou3d_gpu(teacher_box_preds, student_box_preds) # [Nt, Ns]
            iou_3d -= not_same_class # iou < 0 if not from the same class
            matched_iou_of_stduent, matched_teacher_index_of_student = iou_3d.max(0) # [Ns]
            MATCHED_IOU_TH = 0.7
            matched_teacher_mask = (matched_iou_of_stduent >= MATCHED_IOU_TH).float().unsqueeze(-1)
            num_matched_boxes = matched_teacher_mask.sum()
            if num_matched_boxes == 0: num_matched_boxes = 1

        matched_teacher_preds = teacher_box_preds[matched_teacher_index_of_student]
        matched_teacher_cls = teacher_cls_preds[matched_teacher_index_of_student]

        student_box_reg, student_box_rot = student_box_preds[:, :6], student_box_preds[:, [6]]
        matched_teacher_reg, matched_teacher_rot = matched_teacher_preds[:, :6], matched_teacher_preds[:, [6]]

        box_loss_reg = F.smooth_l1_loss(student_box_reg, matched_teacher_reg, reduction='none')
        box_loss_reg = (box_loss_reg * matched_teacher_mask).sum() / num_matched_boxes
        box_loss_rot = F.smooth_l1_loss(torch.sin(student_box_rot - matched_teacher_rot), torch.zeros_like(student_box_rot), reduction='none')
        box_loss_rot = (box_loss_rot * matched_teacher_mask).sum() / num_matched_boxes
        consistency_box_loss = box_loss_reg + box_loss_rot
        consistency_cls_loss = F.smooth_l1_loss(student_cls_preds, matched_teacher_cls, reduction='none')
        consistency_cls_loss = (consistency_cls_loss * matched_teacher_mask).sum() / num_matched_boxes

        box_losses.append(consistency_box_loss)
        cls_losses.append(consistency_cls_loss)
        batch_normalizer += 1

    return sum(box_losses)/batch_normalizer, sum(cls_losses)/batch_normalizer


class SelfEMA(object):
    def __init__(self, model, alpha_first):
        self.last_epoch = model
        self.last_epoch.eval()
        self.alpha_first = alpha_first
        self.alpha = 0
        self.cls_loss_function = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
    
    def get_soft_loss(self, orig_batch_dict, pred_batch_dict):
        '''
            Now we use the OpenPCDet version normalize cls loss.
        '''
        load_data_to_gpu(orig_batch_dict)
        _ =  self.last_epoch(orig_batch_dict)
        last_cls   = orig_batch_dict['batch_cls_preds']
        now_cls    = pred_batch_dict['batch_cls_preds']
        now_box    = pred_batch_dict['batch_box_preds']
        gt_boxes   = orig_batch_dict['gt_boxes']
        batch_size = now_cls.shape[0]
        for i in range(batch_size):
            current_cls_preds = now_cls[i]
            current_box_preds = now_box[i]
            lass_cls_preds    = last_cls[i]
            gt_temp_box       = gt_boxes[i][:, :7]
            gt_temp_class     = gt_boxes[i][:, 7:]

            current_centers, current_size, current_rot = current_box_preds[:, :3], current_box_preds[:, 3:6], current_box_preds[:, 6:]
            gt_centers, gt_size, gt_rot = gt_temp_box[:, :3], gt_temp_box[:, 3:6], gt_temp_box[:, 6:]

            with torch.no_grad():
                current_class = torch.max(current_cls_preds, dim=-1, keepdim=True)[1]
                not_same_class= (gt_temp_class != current_class.T).float()
                MAX_DISTANCE = 1000000
                dist = current_centers[:, None, :] - gt_centers[:, None, :]
                dist = (dist ** 2).sum(-1)
                dist += not_same_class * MAX_DISTANCE

                
            

            

        

        
        
        return now_cls
    
    def update(self, model, epoch, total_epoch):
        self.last_epoch = model
        self.last_epoch.eval()
        self.alpha = self.alpha_first * ((epoch + 1) / total_epoch)
        self.alpha = max(0, self.alpha)





def train_one_epoch_self(model, ema, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        orig_batch = copy.deepcopy(batch)
        

        # get knowledge distillation loss...
        loss, tb_dict, disp_dict = model_func(model, batch)
        soft_loss = ema.get_soft_loss(orig_batch, batch)
        loss = loss + soft_loss * 2

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter



def train_model_selfkd(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, model_ema=None,train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    ema = SelfEMA(model, optim_cfg.START_ALPHA)
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch_self(
                model, ema,optimizer, train_loader, model_func, 
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )
            ema.update(model, cur_epoch, total_epochs)

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                ckpt_ema_name = ckpt_save_dir / ('checkpoint_ema_epoch_%d' % trained_epoch)
                if model_ema is not None:
                    save_ema_checkpoint(model, filename=ckpt_ema_name)






def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def save_ema_checkpoint(state, filename='checkpoint_ema'):
    filename =  '{}.pth'.format(filename )
    torch.save(state, filename)