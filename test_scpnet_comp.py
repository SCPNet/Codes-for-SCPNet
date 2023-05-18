# -*- coding:utf-8 -*-
# author: Xinge, Xzy
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm

# from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")
import yaml


def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

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

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    model_load_path += 'iou26.6891_epoch19.pth'
    if os.path.exists(model_load_path):
        print('Load model from: %s' % model_load_path)
        my_model = load_checkpoint(model_load_path, my_model)
    else:
        print('No existing model, training model from scratch...')

    my_model.to(pytorch_device)

    _, test_dataset_loader, test_pt_dataset = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_tta=True,
                                                                  use_multiscan=True)

    # training
    dataset_name = val_dataloader_config["imageset"]
    output_path = 'out_scpnet/' + dataset_name

    if True:
        print('Generate predictions for test split')
        pbar = tqdm(total=len(test_dataset_loader))
        time.sleep(10)
        ### learning map
        with open("config/label_mapping/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        # make lookup table for mapping
        learning_map_inv = semkittiyaml["learning_map_inv"]
        maxkey = max(learning_map_inv.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut_First = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut_First[list(learning_map_inv.keys())] = list(learning_map_inv.values())

        if True:
            if True:
                my_model.eval()
                with torch.no_grad():
                    for i_iter_test, (_, _, test_grid, _, test_pt_fea, test_index, origin_len) in enumerate(
                            test_dataset_loader):
                        
                        test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          test_pt_fea]
                        test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]

                        predict_labels = my_model(test_pt_fea_ten, test_grid_ten, val_batch_size, test_grid, use_tta=False)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        if True:
                            test_pred_label = np.squeeze(predict_labels)

                            ### save prediction after remapping
                            pred = test_pred_label
                            pred = pred.astype(np.uint32)
                            pred = pred.reshape((-1))
                            upper_half = pred >> 16  # get upper half for instances
                            lower_half = pred & 0xFFFF  # get lower half for semantics
                            lower_half = remap_lut_First[lower_half]  # do the remapping of semantics
                            pred = (upper_half << 16) + lower_half  # reconstruct full label
                            pred = pred.astype(np.uint32)
                            final_preds = pred.astype(np.uint16)
                            
                            save_dir = test_pt_dataset.im_idx[test_index[0]]
                            _,dir2 = save_dir.split('/sequences/',1)
                            new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne', 'predictions')[:-3]+'label'
                            if not os.path.exists(os.path.dirname(new_save_dir)):
                                try:
                                    os.makedirs(os.path.dirname(new_save_dir))
                                except OSError as exc:
                                    if exc.errno != errno.EEXIST:
                                        raise
                            final_preds.tofile(new_save_dir)

                        pbar.update(1)
                del test_grid, test_pt_fea, test_grid_ten, test_index
        pbar.close()
        print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % output_path)
        print('Remapping script can be found in semantic-kitti-api.')

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti-multiscan.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
