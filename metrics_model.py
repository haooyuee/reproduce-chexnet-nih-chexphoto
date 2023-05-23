import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import pandas as pd
import cxr_dataset as CXR
from metrics import Metrics
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
import json
from tqdm import tqdm



def make_metrics_multilabel(data_transforms, model, cfg):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader
    if 'chexphoto' in cfg.name: #load chexphoto
        dataset = CXR.CPDataset(
        path_to_images=cfg.dataset_path,
        path_to_csv=cfg.csv_path,
        fold='val',
        transform=data_transforms['val'])
         
        dataloader = torch.utils.data.DataLoader(
            dataset, BATCH_SIZE, shuffle=False, num_workers=2)
        size = len(dataset)
    
        # create empty dfs
        pred_df = pd.DataFrame(columns=["Path"])
        true_df = pd.DataFrame(columns=["Path"])

    
    else: ##OR load NIH
        dataset = CXR.CXRDataset(
            path_to_images=cfg.dataset_path,
            fold="test",
            transform=data_transforms['val'])
        
        dataloader = torch.utils.data.DataLoader(
            dataset, BATCH_SIZE, shuffle=False, num_workers=2)
        size = len(dataset)
    
        # create empty dfs
        pred_df = pd.DataFrame(columns=["Image Index"])
        true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    t = tqdm(dataloader)
    for i, data in enumerate(t):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow[dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)
    
    metric = Metrics(
        num_classes=len(list(pred_df)[1:]), 
        names=list(pred_df)[1:], 
        threshold=np.zeros((len(list(pred_df)[1:]))) + 0.5,
        true= true_df[true_df.columns.drop('Image Index')].to_numpy(), 
        pred= pred_df[pred_df.columns.drop('Image Index')].to_numpy()
    )
    metric_res = {}
    for key, item in metric.metrics().items():
        print(key)
        metric_res[key] = item()
        #if(i % 10 == 0):
        #    print(str(i * BATCH_SIZE))
    
    pred_df.to_csv(f"results/{cfg.name}_preds.csv", index=False)
    
    json_str = json.dumps(metric_res, indent=4)
    with open(f'results/{cfg.name}_metrics.json', 'w') as json_file:
        json_file.write(json_str)
    #auc_df.to_csv(f"results/{cfg.name}_aucs.csv", index=False)
    return pred_df, metric_res
