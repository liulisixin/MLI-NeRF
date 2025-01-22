'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import argparse
import os

import imaginaire.config
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print, is_master
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.trainers.utils.logging import init_logging
from imaginaire.trainers.utils.get_trainer import get_trainer
from imaginaire.utils.set_random_seed import set_random_seed
from imaginaire.config import AttrDict


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.', default=None)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--show_pbar', action='store_true')
    parser.add_argument('--wandb', action='store_true', help="Enable using Weights & Biases as the logger")
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inference_mode', type=str, default='image')   # image_{dataset} or video_{dataset}_{idx1}_{idx2}
    parser.add_argument('--anno', type=str, default=None)

    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    if args.checkpoint is None:
        logdir = os.path.join('logs', os.path.basename(args.config).split('.yaml')[0])
        checkpoint_file = open(os.path.join(logdir, "latest_checkpoint.txt")).read().strip()
        args.checkpoint = os.path.join(logdir, checkpoint_file)
        args.logdir = os.path.join(logdir, 'output')
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    # special
    # cfg.model.light_visibility = AttrDict(enabled=True, sample_tolerance=32)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Training with {get_world_size()} GPUs.")

    # set random seed by rank
    set_random_seed(args.seed, by_rank=True)

    # Global arguments.
    imaginaire.config.DEBUG = args.debug

    # Create log directory for storing training results.
    cfg.logdir = init_logging(args.config, args.logdir, makedir=True)

    # Print and save final config
    if is_master():
        cfg.print_config()
        cfg.save_config(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    trainer = get_trainer(cfg, is_inference=True, seed=args.seed)
    trainer.checkpointer.load(args.checkpoint, args.resume, load_sch=False, load_opt=False)

    inference_mode = args.inference_mode.split('_')

    split_mode = inference_mode[1]

    cfg.data.preload = True
    cfg.data.train.pseudo_label.enabled = False
    if split_mode == 'train':
        trainer.set_data_loader(cfg, split='train')
        dataloader = trainer.train_data_loader
    elif split_mode == 'test':
        anno = args.anno
        cfg.data.val.annotation = os.path.join(os.path.dirname(cfg.data.val.annotation), anno)
        cfg.data.val.subset = 0
        # cfg.data.val.use_light_index = [0]
        trainer.set_data_loader(cfg, split='val')
        dataloader = trainer.eval_data_loader
    else:
        raise NotImplementedError

    output_folder = trainer.cfg.logdir + '_' + inference_mode[0]
    os.makedirs(output_folder, exist_ok=True)
    if inference_mode[0] == 'image':
        # trainer.test(dataloader, mode="test",
        #              output_dir=output_folder, show_pbar=args.show_pbar)
        trainer.test_save(dataloader, mode="test",
                     output_dir=output_folder, show_pbar=args.show_pbar)
    elif inference_mode[0] == 'rene':
        # setting_list = ['c25l10', 'c25l15', 'c25l19', 'c25l8']
        setting_list = ['c25l10']
        trainer.test_images(dataloader, output_dir=output_folder, setting_list=setting_list, mode="test",
                            show_pbar=args.show_pbar)
    elif inference_mode[0] == 'video':
        setting1 = inference_mode[2]
        setting2 = inference_mode[3]
        if hasattr(trainer.cfg.model.object.rgb, 'network_mode'):
            network_mode = trainer.cfg.model.object.rgb.network_mode
            length_output = len(network_mode.split('_'))
        else:
            length_output = 1
        if length_output == 3:
            video_content = ('rgb', 'gt', 'o_r', 'o_s', 'o_re')
        elif length_output == 2:
            video_content = ('rgb', 'gt', 'o_r', 'o_s')
        else:
            video_content = ('rgb', 'gt')
        trainer.test_video(dataloader, setting1, setting2,
                           mode="test",
                           output_dir=output_folder, show_pbar=args.show_pbar,
                           video_content=video_content)
    elif inference_mode[0] == 'lights':
        # prepare for pseudo labels; For datasets like ReNe; With grid-like sampling of cameras and lights.
        trainer.test_all_light(dataloader,
                               output_dir=output_folder,
                               mode="test",
                               dataset_type='pair')
    elif inference_mode[0] == 'unpairlights':
        # prepare for pseudo labels; For datasets like synthetic datasets, cameras and lights are randomly sampled.
        trainer.test_all_light(dataloader,
                               output_dir=output_folder,
                               mode="test",
                               dataset_type='unpair',
                               sample_num=4, seed=999)
    elif inference_mode[0] == 'limitedlights':
        # prepare for pseudo labels; For datasets like synthetic datasets (NL1, NL4), the number of lights is limited.
        trainer.test_all_light(dataloader,
                               output_dir=output_folder,
                               mode="test",
                               dataset_type='limitedlights',
                               sample_num=4, seed=999)
    elif inference_mode[0] == 'singlelight':
        # prepare for pseudo labels; For datasets like synthetic datasets (NL1, NL4), the number of lights is limited.
        trainer.test_all_light(dataloader,
                               output_dir=output_folder,
                               mode="test",
                               dataset_type='limitedlights',
                               sample_num=1, seed=999)

    pass

if __name__ == "__main__":
    main()
