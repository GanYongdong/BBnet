import argparse
import logging
import os

import torch
import torch.distributed as dist

from net.utils import dist_util
from net.utils.misc import str2bool
from net.utils.dist_util import synchronize
from net.config import cfg
from net.utils.misc import mkdir
from net.utils.logger import setup_logger

from net import BBnet

def train(cfg, args):
    logger = logging.getLogger('NET.trainer')
    # model = build_detection_model(cfg)

    print("==========> Loading datasets")
    
    print("==========> Building model")
    model = BBnet()
    
    print('train finish')

def main():
    print('hello main')
    
    parser = argparse.ArgumentParser(description='net')
    parser.add_argument(
        "--config-file", default="configs/vgg_ssd300_voc0712.yaml",
        metavar="FILE", help="path to config file", type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--skip-test", dest="skip_test", 
        help="Do not test the final model", action="store_true",
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("NETt", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()

    print('finish main')

if __name__ == '__main__':
    main()
