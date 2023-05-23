#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on 2023-0119$

@author: Jonathan Beaulieu-Emond
"""


import numpy as np
import sklearn.metrics
#import timm.utils.metrics
from sklearn import metrics
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

import logging


class Metrics:
    def __init__(self, num_classes, names, threshold):
        self.num_classes = num_classes
        self.thresholds = threshold
        self.names = names

        self.convert = lambda x: x
       
    def perf_measure(self, true, pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(self.names)):
            if true[i]==pred[i]==1:
                TP += 1
            if pred[i]==1 and true[i]!=pred[i]:
                FP += 1
            if true[i]==pred[i]==0:
                TN += 1
            if pred[i]==0 and true[i]!=pred[i]:
                FN += 1

        return(TP, FP, TN, FN)

    ###################################################################
    def accuracy(self, true, pred, threshold):
       
        #We store the threshold values from a default list or from the parameter threshold
        if len(threshold) != len(self.names):      
            thresholdArray = np.array(list(self.thresholds.values()))
        else:
            thresholdArray = np.array(list(threshold.values()))
           
        #We do not want to change the reference content provided
        pred2 = np.copy(pred)
        pred2 = self.convert(pred2)
       
        for idx, thresh in enumerate(thresholdArray):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
         
        accuracy = 0
        for x, y in zip(true, pred2):
            if (x == y).all():
                accuracy += 1
           

        return accuracy

    ###################################################################
    def f1(self, true, pred, threshold):
       
        if len(threshold) != len(self.names):      
            thresholdArray = np.array(list(self.thresholds.values()))
        else:
            thresholdArray = np.array(list(threshold.values()))
       
        #We do not want to change the reference content provided in the input parameters
        pred2 = np.copy(pred)
       
        for idx, thresh in enumerate(thresholdArray):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)

        f1 = f1_score(true, pred2, zero_division=0, average=None)
        f1Globale = f1_score(true, pred2, average='weighted')
       
        f1_dict = {name: item for name, item in zip(self.names, f1)}
        f1_dict["mean"] = np.mean(f1)
       
        f1_dict["globale"] = f1Globale
       
        return f1_dict

    ###################################################################
    def precision(self, true, pred, threshold):
       
        if len(threshold) != len(self.names):      
            thresholdArray = np.array(list(self.thresholds.values()))
        else:
            thresholdArray = np.array(list(threshold.values()))

        #We do not want to change the reference content provided
        pred2 = np.copy(pred)
        pred2 = self.convert(pred2)
       
        #pred = np.where(pred > 0.5, 1, 0)
        for idx, name in enumerate(self.names):
            pred2[:,idx] = np.where(pred2[:,idx] > threshold[name], 1, 0)
       
        results = precision_score(true, pred2, average=None, zero_division=0)
        precisionGlobale = precision_score(true, pred2, average='weighted', zero_division=0)

        results_dict = {}
        for item, name in zip(results, self.names):
            results_dict[name] = item
           
        results_dict["globale"] = precisionGlobale
       
        return results_dict

    ###################################################################
    def recall(self, true, pred, threshold):

        #We do not want to change the reference content provided
        pred2 = np.copy(pred)
        pred2 = self.convert(pred2)
       
        #pred = np.where(pred > 0.5, 1, 0)
        if len(threshold) != len(self.names):      
            thresholdArray = np.array(list(self.thresholds.values()))
        else:
            thresholdArray = np.array(list(threshold.values()))
           
        for idx, thresh in enumerate(thresholdArray):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
       
        results = recall_score(true, pred2, average=None, zero_division=0)
        recallGlobale = recall_score(true, pred2, average='weighted', zero_division=0)
       
        results_dict = {}
        for item, name in zip(results, self.names):
            results_dict[name] = item
           
        results_dict['globale'] = recallGlobale
       
        return results_dict

    ###################################################################
    def computeAUROC(self, true, pred, threshold):

        fpr = dict()
        tpr = dict()
        outAUROC = dict()
        classCount = pred.shape[1]
       
         #We do not want to change the reference content provided
        pred2 = np.copy(pred)
        pred2 = self.convert(pred2)
 
        for i in range(classCount):

            # fpr[i], tpr[i], thresholds = roc_curve(true[:, i], pred[:, i],pos_label=1)
            #
            # threshold = thresholds[np.argmax(tpr[i] - fpr[i])]
            # logging.info(f"threshold {self.names[i]} : ",threshold)
            # self.thresholds[i] =threshold
            # try :
            #     auroc =  auc(fpr[i], tpr[i])
            # except :
            #     auroc=0
            try:
                #Modif JFS on Feb 14th 2023, before average='macro' instead of multi_class='ovr'
                auroc = roc_auc_score(true[:, i], pred2[:, i], multi_class='ovr')
            except ValueError:
                auroc = 0
           
            outAUROC[self.names[i]] = auroc
            if np.isnan(outAUROC[self.names[i]]):
                outAUROC[self.names[i]] = 0

       
        outAUROC["mean"] = np.mean(list(outAUROC.values()))

        return outAUROC

    ###################################################################
    def mmc(self, true, pred, threshold):
       
        #We do not want to change the reference content provided
        pred2 = np.copy(pred)
        pred2 = self.convert(pred2)
       
       
        if len(threshold) != len(self.names):      
            thresholdArray = np.array(list(self.thresholds.values()))
        else:
            thresholdArray = np.array(list(threshold.values()))
   
        for idx, thresh in enumerate(thresholdArray):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
       
        # Convertir les étiquettes en un format binaire
        #y_true_bin = label_binarize(true, classes=range(self.num_classes))
       
        results = {}
        mmc_mean = 0.0
        for i in range(self.num_classes):
            results[self.names[i]] = matthews_corrcoef(true[:, i], pred2[:, i])
            mmc_mean += results[self.names[i]]

        # Calculer le MCC moyen (moyenne macro)
        results['mean'] = mmc_mean/self.num_classes

        return results

    ###################################################################
    def g_mean(self, true, pred, threshold):
       
        #We do not want to change the reference content provided
        pred2 = np.copy(pred)
        pred2 = self.convert(pred2)
       
        if len(threshold) != len(self.names):      
            thresholdArray = np.array(list(self.thresholds.values()))
        else:
            thresholdArray = np.array(list(threshold.values()))
   
        for idx, thresh in enumerate(thresholdArray):
            pred2[:, idx] = np.where(pred2[:, idx] > thresh, 1, 0)
       
        g_mean={}

        #Initialize the g_mean
        for i in range(self.num_classes):
            g_mean[self.names[i]] = 0.0
       
        #Check if we can calculate gmean    
        if true.shape != pred2.shape or true.shape[0] == pred2.shape[0] == 1:
            return g_mean
 
        for i in range(self.num_classes):
            tn, fp, fn, tp = confusion_matrix(true[:, i], pred2[:, i]).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            g_mean[self.names[i]] = np.sqrt(specificity * sensitivity)

        return g_mean
   
    ###################################################################
    def metrics(self):
        dict = {
            "auc":      self.computeAUROC,
            "f1":       self.f1,
            "recall":   self.recall,
            "MMC" :     self.mmc, #TODO : Fix MMC for multi-label
            "precision":self.precision,
            "accuracy": self.accuracy,
            "gmean":    self.g_mean,
        }
        return dict

   
###################################################################
if __name__ == "__main__":
    #from radia import names
    np.random.seed(0)
    names = ['a','b','c','d','e']
    num_classes = len(names)
    metric = Metrics(
        num_classes=num_classes, names=names, threshold=np.zeros((num_classes)) + 0.5
    )
    res = {}
    threshold = {}
    for idx, name in enumerate(metric.names):
        threshold[name] = 0.5
    
    metrics = metric.metrics()
    #print(metrics)
    label = np.random.randint(0, 2, (10, num_classes))
    pred = np.random.random(size=(10, num_classes))
    for key, metric in metrics.items():
        res[key] = metric(label, pred, threshold)
    print(res)
