import argparse
import datetime
import mayavi.mlab as mlab
from pathlib import Path




from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


from visual_utils import visualize_utils as V

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')



    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()


    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)



    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=1,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=10
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    for idx, data in enumerate(train_loader):
        gt_boxes = data['gt_boxes'].squeeze()
        points   = data['points']
        gt_boxes = gt_boxes[:, :7]
        load_data_to_gpu(data)
        pred_dicts, _ = model.forward(data)
        pred_boxes = pred_dicts[0]['pred_boxes'].detach()
        pred_score = pred_dicts[0]['pred_scores'].detach()
        pred_label = pred_dicts[0]['pred_labels'].detach()
        indice     = pred_score > 0.4
        pred_boxes = pred_boxes[indice]
        pred_score = pred_score[indice]
        pred_label = pred_label[indice]
        V.draw_scenes(
            points=points[:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_boxes,
            ref_scores=pred_score, ref_labels=pred_label
        )
        mlab.show(stop=True)
    


if __name__ == '__main__':
    main()