'''This file implement the knowledge distillation for models...'''
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from easydict import EasyDict

from pcdet.config import cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, teacher_model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.kd_train_utils import train_model_kd

student_cfg = EasyDict()
student_cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
student_cfg.LOCAL_RANK = 0

teacher_cfg = EasyDict()
teacher_cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
teacher_cfg.LOCAL_RANK = 0


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_student_file', type=str, default=None, help='specify the student config for training')
    parser.add_argument('--cfg_teacher_file', type=str, default=None, help='specify the teacher config for training')
    parser.add_argument('--teacher_model', type=str, default=None, help='pretrained teacher model')
    parser.add_argument('--pretrain_student_model', type=str, default=None, help='pretrained student model')


    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=40, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_student_file, student_cfg)
    student_cfg.TAG = Path(args.cfg_student_file).stem
    student_cfg.EXP_GROUP_PATH = '/'.join(args.cfg_student_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'


    cfg_from_yaml_file(args.cfg_teacher_file, teacher_cfg)
    teacher_cfg.TAG = Path(args.cfg_teacher_file).stem
    teacher_cfg.EXP_GROUP_PATH = '/'.join(args.cfg_teacher_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    return args, student_cfg, teacher_cfg


def main():
    args, student_cfg, teacher_cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, student_cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = student_cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = student_cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = student_cfg.ROOT_DIR / 'output' / student_cfg.EXP_GROUP_PATH / student_cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=student_cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(student_cfg, logger=logger)
    if student_cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_student_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if student_cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=student_cfg.DATA_CONFIG,
        class_names=student_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    model = build_network(model_cfg=student_cfg.MODEL, num_class=len(student_cfg.CLASS_NAMES), dataset=train_set)
    teacher_model = build_network(model_cfg=teacher_cfg.MODEL, num_class=len(student_cfg.CLASS_NAMES), dataset=train_set)
    for params in teacher_model.parameters():
        params.detach_()

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
    model.cuda()
    teacher_model.cuda()


    optimizer = build_optimizer(model, student_cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.teacher_model is not None:
        teacher_model.load_params_from_file(filename=args.teacher_model, to_cpu=dist_train, logger=logger)
    else:
        logger.info("The distillation process need the pretrained teacher models")
        exit()

    if args.pretrain_student_model is not None:
        model.load_params_from_file(filename=args.pretrain_student_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    teacher_model.eval()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[student_cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)
    logger.info("Student model:::::")
    logger.info(model)
    logger.info("Teacher model:::::")
    logger.info(teacher_model)


    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=student_cfg.OPTIMIZATION
    )
    logger.info("Init finish.....")

    logger.info('**********************Start training %s/%s(%s)**********************'
                % (student_cfg.EXP_GROUP_PATH, student_cfg.TAG, args.extra_tag))
    train_model_kd(
        model,
        teacher_model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        teacher_model_fn_decorator=teacher_model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=student_cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=student_cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (student_cfg.EXP_GROUP_PATH, student_cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (student_cfg.EXP_GROUP_PATH, student_cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=student_cfg.DATA_CONFIG,
        class_names=student_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (student_cfg.EXP_GROUP_PATH, student_cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
