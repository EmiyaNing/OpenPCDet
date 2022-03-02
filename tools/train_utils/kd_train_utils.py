import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

def compact_batch(tensor_list):
    '''
        Write some code to pad the teacher predicts....
    '''
    bs, ch, fix = tensor_list[0].shape
    max_ch = ch
    pad_tensor_list = []
    for tensor in tensor_list:
        if max_ch < tensor.shape[1]:
            max_ch = tensor.shape[1]

    for tensor in tensor_list:
        pad_tensor = torch.zeros(bs, max_ch - tensor.shape[1], fix).cuda()
        tensor = torch.cat([tensor, pad_tensor], dim=1)
        pad_tensor_list.append(tensor)
    
    paded_tensor = torch.cat(pad_tensor_list, dim=0)
    return paded_tensor

def update_ema_variables(model, ema_model, global_step):
    alpha = min(1 - 1 / (global_step + 1), 0.999)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train_one_epoch_sess(model, model_ema, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
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
        model_ema.train()
        optimizer.zero_grad()
        # ema_model forward........
        batch['is_ema'] = True
        ema_dicts = model_func(model, batch)
        batch['is_ema'] = False        
        batch['ema_cls_preds'] = ema_dicts['ema_cls_preds']
        batch['ema_box_preds'] = ema_dicts['ema_box_preds']
        batch['ema_features']  = ema_dicts['ema_feature']
        # get knowledge distillation loss...
        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        # Update the ema_models...
        update_ema_variables(model, model_ema, accumulated_iter)

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


def train_one_epoch(model, teacher_model, optimizer, train_loader, model_func, teacher_model_fn_decorator, lr_scheduler, accumulated_iter, optim_cfg,
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
        teacher_model.eval()
        optimizer.zero_grad()
        # get teacher predict result...
        # use teacher predict as hard label to caculate the loss....
        predict_dicts, recall_dicts = teacher_model_fn_decorator(teacher_model, batch)
        boxes  = [dict['pred_boxes'].unsqueeze(0) for dict in predict_dicts]
        labels = [dict['pred_labels'].unsqueeze(-1).unsqueeze(0) for dict in predict_dicts]
        pad_boxes  = compact_batch(boxes)
        pad_labels = compact_batch(labels)
        batch['teacher_box'] = torch.cat([pad_boxes, pad_labels], dim=-1)
        batch_box_preds = [dict['teacher_box_preds'].unsqueeze(0) for dict in predict_dicts]
        batch_cls_preds = [dict['teacher_cls_preds'].unsqueeze(0) for dict in predict_dicts]
        batch_teacher_feature = [dict['teacher_feature'].unsqueeze(0) for dict in predict_dicts]
        batch_teacher_cls_temp = [dict['teacher_cls_temp'].unsqueeze(0) for dict in predict_dicts]
        batch_teacher_reg_temp = [dict['teacher_reg_temp'].unsqueeze(0) for dict in predict_dicts]
        teacher_box_preds = torch.cat(batch_box_preds, dim=0)
        teacher_cls_preds = torch.cat(batch_cls_preds, dim=0)
        teacher_feature   = torch.cat(batch_teacher_feature, dim=0)
        head_cls_temp     = torch.cat(batch_teacher_cls_temp, dim=0)
        head_reg_temp     = torch.cat(batch_teacher_reg_temp, dim=0)
        batch['teacher_cls_preds'] = teacher_cls_preds
        batch['teacher_box_preds'] = teacher_box_preds
        batch['teacher_feature'] = teacher_feature
        batch['teacher_head_cls'] = head_cls_temp
        batch['teacher_head_reg'] = head_reg_temp
        

        # get knowledge distillation loss...
        loss, tb_dict, disp_dict = model_func(model, batch)

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


def train_model_kd(model, teacher_model, optimizer, train_loader, model_func, teacher_model_fn_decorator,lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
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
            accumulated_iter = train_one_epoch(
                model, teacher_model, optimizer, train_loader, model_func, teacher_model_fn_decorator,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

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
