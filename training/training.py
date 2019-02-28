# coding='utf-8'
# Необходимо в загрузку из COCO добавить использование class ImageBaseAug(object) для улучшения обуччения
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import prepare_train_dataset, prepare_val_dataset
#from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
#from common.coco_dataset import COCODataset

device_run= torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 16,
    "train_path": "../data/coco/trainvalno5k.txt",
    "epochs": 100,
    "img_h": 416,
    "img_w": 416,
    "parallels": [0], #[0,1,2,3],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    "pretrain_snapshot": "weights/official_yolov3_weights_pytorch.pth",                        #  load checkpoint
    "evaluate_type": "", 
    "try": 0,
    "export_onnx": False,
}


def train(config):
    
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    #if len(config["parallels"])>1:
    net = nn.DataParallel(net).to(device_run)
  #  else:
    #    net=  net.to(device_run)
    #net = net.cuda()
    

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"],map_location=device_run)
        net.load_state_dict(state_dict)
    
    net.train(is_training)

    # Only export onnx
    # if config.get("export_onnx"):
        # real_model = net.module
        # real_model.eval()
        # dummy_input = torch.randn(8, 3, config["img_h"], config["img_w"]).cuda()
        # save_path = os.path.join(config["sub_working_dir"], "pytorch.onnx")
        # logging.info("Exporting onnx to {}".format(save_path))
        # torch.onnx.export(real_model, dummy_input, save_path, verbose=False)
        # logging.info("Done. Exiting now.")
        # sys.exit()

    # Evaluate interface
    # if config["evaluate_type"]:
        # logging.info("Using {} to evaluate model.".format(config["evaluate_type"]))
        # evaluate_func = importlib.import_module(config["evaluate_type"]).run_eval
        # config["online_net"] = net

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"]),device_run))

    # DataLoader
    #dataloader = torch.utils.data.DataLoader(COCODataset(config["train_path"],
    #                                                     (config["img_w"], config["img_h"]),
    #                                                     is_training=True),
    #                                         batch_size=,
    #                                         shuffle=True, num_workers=32, pin_memory=True)

    train_img_datasets, train_dataloader = prepare_train_dataset('coco', config["img_w"], config["batch_size"])
    #val_img_datasets, val_dataloder = prepare_val_dataset(
    #    args.dataset, args.reso, args.bs, seq=args.seq)

    # Start the training loop
    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        for step, (filename,images, labels) in enumerate(train_dataloader):
            #images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["global_step"] += 1

            # Forward and backward
            optimizer.zero_grad()
            outputs = net(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], list(labels))
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            loss.backward()
            optimizer.step()

            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f "%
                    (epoch, step, _loss, example_per_second, lr)
                )
                config["tensorboard_writer"].add_scalar("lr",
                                                        lr,
                                                        config["global_step"])
                config["tensorboard_writer"].add_scalar("example/sec",
                                                        example_per_second,
                                                        config["global_step"])
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    config["tensorboard_writer"].add_scalar(name,
                                                            value,
                                                            config["global_step"])

            #if step > 0 and step % 1000 == 0:
                # net.train(False)
               # _save_checkpoint(net.state_dict(), config)
                # net.train(True)

        lr_scheduler.step()

    # net.train(False)
  #  _save_checkpoint(net.state_dict(), config)
    # net.train(True)
    logging.info("Bye~")

# best_eval_result = 0.0
def _save_checkpoint(state_dict, config, evaluate_func=None):
    # global best_eval_result
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)
    # eval_result = evaluate_func(config)
    # if eval_result > best_eval_result:
        # best_eval_result = eval_result
        # logging.info("New best result: {}".format(best_eval_result))
        # best_checkpoint_path = os.path.join(config["sub_working_dir"], 'model_best.pth')
        # shutil.copyfile(checkpoint_path, best_checkpoint_path)
        # logging.info("Best checkpoint saved to {}".format(best_checkpoint_path))
    # else:
        # logging.info("Best result: {}".format(best_eval_result))


def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    config = TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    #config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    train(config)

if __name__ == "__main__":
    main()
