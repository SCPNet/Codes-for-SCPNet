# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py 

import torch


# def load_checkpoint_old2(model_load_path, model):
def load_checkpoint(model_load_path, model):
    pre_weight = torch.load(model_load_path)
    my_model_dict = model.state_dict()
    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        # str3 = 'seg_head.sparseModel.1.weight'
        # if k.find(str3) > 0:
        #     value = value[:, :, 0, :]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
            match_size += 1
            part_load[k] = value
        else:
            # print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
            Lvalue = len(value.shape)
            assert 1 <= Lvalue <= 5
            if len(value.shape) == 1:
                c = value.shape[0]
                cc = my_model_dict[k].shape[0] - c #int(c*0.5)
                if 0 < cc <= c:
                    value = torch.cat([value, value[:cc]], dim=0)
                elif cc > c:
                    value = torch.cat([value, value, value[:(cc-c)]], dim=0)
                elif cc < 0:
                    value = value[:-cc]
            else:
                # print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
                cs = value.shape
                ccs = [0]*Lvalue
                j = -1
                for ci in cs:
                    j += 1
                    ccs[j] = (my_model_dict[k].shape[j] - ci)
                    if ccs[j] != 0:
                        for m in range(Lvalue):
                            if m != j:
                                ccs[m] = value.shape[m]
                        # print(ccs)
                        if ccs[j] > 0:
                            if ccs[j] > ci:
                                ccs[j] = ccs[j] - ci
                            if Lvalue == 5:
                                value = torch.cat([value, value[:ccs[0], :ccs[1], :ccs[2], :ccs[3], :ccs[4]]], dim=j)
                            elif Lvalue == 4:
                                # print(value[:ccs[0], :ccs[1], :ccs[2], :ccs[3]].shape)
                                value = torch.cat([value, value[:ccs[0], :ccs[1], :ccs[2], :ccs[3]]], dim=j)
                            elif Lvalue == 3:
                                value = torch.cat([value, value[:ccs[0], :ccs[1], :ccs[2]]], dim=j)
                            elif Lvalue == 2:
                                value = torch.cat([value, value[:ccs[0], :ccs[1]]], dim=j)
                            ccs[j] = value.shape[j]
                        elif ccs[j] < 0:
                            # ccs[j] = -ccs[j]
                            ccs[j] = my_model_dict[k].shape[j]
                            if j == 0:
                                value = value[:ccs[0], :]
                            elif j == 1:
                                if Lvalue == 2:
                                    value = value[:, :ccs[1]]
                                else:
                                    value = value[:, :ccs[1], :]
                            elif j == 2:
                                if Lvalue == 3:
                                    value = value[:, :, :ccs[2]]
                                else:
                                    value = value[:, :, :ccs[2], :]
                            elif j == 3 and j <= Lvalue-1:
                                if Lvalue == 4:
                                    value = value[:, :, :, :ccs[3]]
                                else:
                                    value = value[:, :, :, :ccs[3], :]
                            elif j == 4 and j <= Lvalue-1:
                                value = value[:, :, :, :, :ccs[4]]

            nomatch_size += 1
            if my_model_dict[k].shape == value.shape:
                part_load[k] = value
            # print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
            # assert my_model_dict[k].shape == value.shape

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)
    # model.load_state_dict(my_model_dict, strict=False)  # True

    return model


def load_checkpoint_old2(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
            match_size += 1
            part_load[k] = value
        else:
            assert len(value.shape) == 1 or len(value.shape) == 5
            if len(value.shape) == 1:
                c = value.shape[0]
                cc = my_model_dict[k].shape[0] - c #int(c*0.5)
                if cc <= c:
                    value = torch.cat([value, value[:cc]], dim=0)
                else:
                    value = torch.cat([value, value, value[:(cc-c)]], dim=0)
            else:
                _, _, _, c1, c2 = value.shape
                cc1 = my_model_dict[k].shape[3] - c1 #int(c1*0.5)
                cc2 = my_model_dict[k].shape[4] - c2 #int(c2*0.5)
                if cc1 > 0 and cc1 <= c1:
                    value1 = torch.cat([value, value[:, :, :, :cc1, :]], dim=3) 
                elif cc1 > c1:
                    value1 = torch.cat([value, value, value[:, :, :, :(cc1-c1), :]], dim=3) 
                else:
                    value1 = value
                if cc2 > 0 and cc2 <= c2:
                    value = torch.cat([value1, value1[:, :, :, :, :cc2]], dim=4) 
                elif cc2 > c2:
                    value = torch.cat([value1, value1, value1[:, :, :, :, :(cc2-c2)]], dim=4) 
                else:
                    value = value1
            nomatch_size += 1
            part_load[k] = value
            assert my_model_dict[k].shape == value.shape
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    # model.load_state_dict(my_model_dict)
    model.load_state_dict(my_model_dict, strict=False)  # True

    return model

def load_checkpoint_old(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[k] = value
        else:
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model

def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
