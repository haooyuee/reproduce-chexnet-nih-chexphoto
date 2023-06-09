import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import pandas as pd
import cxr_dataset as CXR
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm



def make_pred_multilabel(data_transforms, model, cfg):
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
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        #if(i % 10 == 0):
        #    print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:
        if 'chexphoto' in cfg.name:
            if column not in [
                'Enlarged Cardiomediastinum',
                'Cardiomegaly',
                'Lung Opacity',
                'Lung Lesion',
                'Edema',
                'Consolidation',
                'Pneumonia',
                'Atelectasis',
                'Pneumothorax',
                'Pleural Effusion',
                'Pleural Other',
                'Fracture',
                'Support Devices']:
                    continue
        else:
            if column not in [
                'Atelectasis',
                'Cardiomegaly',
                'Effusion',
                'Infiltration',
                'Mass',
                'Nodule',
                'Pneumonia',
                'Pneumothorax',
                'Consolidation',
                'Edema',
                'Emphysema',
                'Fibrosis',
                'Pleural_Thickening',
                'Hernia']:
                        continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            #thisrow['auc'] = sklm.roc_auc_score(
            #    actual.as_matrix().astype(int), pred.as_matrix()) #not need as_matrix
            thisrow['auc'] = sklm.roc_auc_score(
                actual.astype(int), pred)    
        except BaseException:
            print("can't calculate auc for " + str(column))
            
        '''   
        thisrow['auc'] = sklm.roc_auc_score(
                actual.astype(int), pred)    
        '''
        auc_df = auc_df.append(thisrow, ignore_index=True)

    pred_df.to_csv(f"results/{cfg.name}_preds.csv", index=False)
    auc_df.to_csv(f"results/{cfg.name}_aucs.csv", index=False)
    return pred_df, auc_df
