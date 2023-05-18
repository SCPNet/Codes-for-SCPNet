# -*- coding:utf-8 -*-
# author: Xinge, Xzy
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name, get_eval_mask, unpack
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings
from utils.np_ioueval import iouEval
import yaml

warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    model_load_path += '0.pth'
    model_save_path += ''
    if os.path.exists(model_load_path):
        print('Load model from: %s' % model_load_path)
        my_model = load_checkpoint(model_load_path, my_model)
    else:
        print('No existing model, training model from scratch...')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    print(model_save_path)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader, val_pt_dataset = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_tta=False,
                                                                  use_multiscan=True)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    # learning map
    with open("config/label_mapping/semantic-kitti.yaml", 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    class_strings = semkittiyaml["labels"]
    class_inv_remap = semkittiyaml["learning_map_inv"]

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # lr_scheduler.step(epoch)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea, train_index, origin_len) in enumerate(train_dataset_loader):
            
            if global_iter % check_iter == 0 and epoch > 0:
                my_model.eval()

                val_loss_list = []
                val_method = 2  # 1-segmentation method, 2-completion method
                if val_method == 1:
                    hist_list = []
                else:
                    evaluator = iouEval(num_class, [])
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, _, val_pt_fea, val_index, origin_len) in enumerate(
                            val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                        val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]

                        for bat in range(val_batch_size):

                            val_label_tensor = val_vox_label[bat,:].type(torch.LongTensor).to(pytorch_device)
                            val_label_tensor = torch.unsqueeze(val_label_tensor, 0)
                            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                                    ignore=ignore_label) + loss_func(predict_labels.detach(), val_label_tensor)

                            predict_labels = torch.argmax(predict_labels, dim=1)
                            predict_labels = predict_labels.cpu().detach().numpy()
                            predict_labels = np.squeeze(predict_labels)
                            val_vox_label0 = val_vox_label[bat, :].cpu().detach().numpy()
                            val_vox_label0 = np.squeeze(val_vox_label0)
                            
                            val_name = val_pt_dataset.im_idx[val_index[0]]
                            
                            invalid_name = val_name.replace('velodyne', 'voxels')[:-3]+'invalid'
                            invalid_voxels = unpack(np.fromfile(invalid_name, dtype=np.uint8))  # voxel labels
                            invalid_voxels = invalid_voxels.reshape((256, 256, 32))
                            masks = get_eval_mask(val_vox_label0, invalid_voxels)
                            predict_labels = predict_labels[masks]
                            val_vox_label0 = val_vox_label0[masks]
                            
                            evaluator.addBatch(predict_labels.astype(int), val_vox_label0.astype(int))
                            
                            val_loss_list.append(loss.detach().cpu().numpy())

                # my_model.train()
                print('Validation per class iou: ')
                _, class_jaccard = evaluator.getIoU()
                m_jaccard = class_jaccard[1:].mean()
                iou = class_jaccard
                val_miou = m_jaccard * 100
                ignore = [0]
                # print also classwise
                for i, jacc in enumerate(class_jaccard):
                    if i not in ignore:
                        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                            i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
                # compute remaining metrics.
                conf = evaluator.get_confusion()
                acc_completion = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0])
                print('Current val completion iou is %.3f' %  acc_completion)
                
                del val_vox_label, val_grid, val_pt_fea, val_pt_fea_ten, val_grid_ten, val_label_tensor

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    # save model with best val miou for completion
                    model_save_name = model_save_path + ('iou%.4f_epoch%d.pth' % (val_miou, epoch))
                    torch.save(my_model.state_dict(), model_save_name)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))

                my_model.train()

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, point_label_tensor.shape[0])
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=ignore_label) + loss_func(
                outputs, point_label_tensor)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti-multiscan.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
