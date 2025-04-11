import torch
import numpy as np
import math
import dataset

def get_cls_info(model):
    list_layer_name = []
    list_layer_shape = []
    for (name, weights) in model.classifier.state_dict().items():
        tokens = name.split('.')
        if tokens[-1] == 'weight':
            list_layer_name.append(name)
            list_layer_shape.append(weights.shape)

    return list_layer_name[:-1], list_layer_shape[:-1]

def get_rows(nb_rows, ratio=0.5):
    points = np.sin(np.linspace(0, 1, int(nb_rows*ratio)))
    rows = np.unique((points * nb_rows).astype(int))

    return rows

def backward_hook(grad):
    nb_rows = grad.shape[0]
    rows = get_rows(nb_rows)

    mask = torch.ones_like(grad)
    mask[rows] = 0

    return grad * mask

def set_hook(model):
    list_layer_name, _ = get_cls_info(model)

    named_params = dict(model.classifier.named_parameters())

    for weight_name in list_layer_name:
        param = named_params[weight_name]
        param.register_hook(backward_hook)

def insert_data_to_model(model, data, layer_name, chunk_size, rows, row_idx):
    is_next_layer = True

    data_flat = data.flatten()
    chunks = torch.split(data_flat, chunk_size)
    len_chunks = len(chunks)

    if len_chunks > (len(rows) - row_idx):
        is_next_layer = False
        return is_next_layer, -1
    
    with torch.no_grad():
        param = dict(model.classifier.named_parameters())[layer_name]
        for chunk in chunks:
            param[rows[row_idx], :chunk.shape[0]] = chunk
            row_idx += 1
            
    return is_next_layer, row_idx

def init_set_model(model, dataset):
    set_hook(model)
    
    list_layer_name, list_layer_shape = get_cls_info(model)

    len_data = len(dataset)
    data_idx = 0
    for layer_idx in range(len(list_layer_name)):
        is_next_layer = True
        layer_name = list_layer_name[layer_idx]
        layer_shape = list_layer_shape[layer_idx]
        rows = get_rows(layer_shape[0])
        chunk_size = layer_shape[1]
        row_idx = 0

        while is_next_layer:
            if data_idx >= len_data:
                return
            
            if dataset[data_idx][1] is 0:
                data_idx += 1
                continue

            data_sample = dataset[data_idx][0]
            is_next_layer, row_idx = insert_data_to_model(model, data_sample, layer_name, chunk_size, rows, row_idx)
            data_idx += 1
        