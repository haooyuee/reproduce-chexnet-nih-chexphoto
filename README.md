# task list：
- Étape #1 : Charger le code de GitHub
Done
- Étape #2 : Vous devez disposer d’un GPU 
3060m and Google Colab
- Étape #3 : charger le training dataset initale: https://www.kaggle.com/datasets/nih-chest-xrays/data
Charger the nih resized dataset : [nih resized 224*224](https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0)
- Étape #4 : Lire le Readme et les documentations
Done
- Étape #5 : Relancer le training et la validation
Retrain_colab.ipynb (Use the computing resources of colab to train)
- Étape #5 : Vérifier la convergence
plot.ipynb
- Étape #6: Try to load the ChexPhoto Dataset : https://stanfordmlgroup.github.io/competitions/chexphoto/
Done (The resized dataset cannot be found, so use the validation set and the test set  as the training set validation set. Thus saving computing resources.)
- Étape #7 : Try to relaunch the training with the ChexPhoto dataset
Retrain_chexphoto.ipynb
- Étape #8 : Vérifier la convergence et comparer les métriques convergences
plot.ipynb

# Answer some questions
- Comprendre où se trouve le data loader, le loss calculator, l'optimizer, Quel est l'algo de loss ?
Generally in pytorch, we first create a Dataset, then use torch.DATALOADER to load the dataset. (in model.py (line 279))

Loss function: Binary Cross Entropy and is used for binary classification, which is a very common choice in multi-task classification. We need to calculate the BCEloss of each label separately and then combine them to obtain the total Loss.

Optimizer: Stochastic gradient descent which is a Basic optimizer, maybe we can try SGD with momentum OR Adam to get better performance.

- Quelles sont les métrics ?
We use the Area Under the Curve (AUC) as the primary evaluation metric for our classification models. In an imbalanced dataset(like medic image), some classes have significantly fewer samples than others, which can lead to biased model performance and misleading evaluation metrics.

Traditional evaluation metrics such as accuracy may not provide an accurate representation of the model's performance in such cases, as a high accuracy can be achieved by merely predicting the majority class. Similarly, the F1-score, which combines precision and recall, can also be affected by the imbalance, resulting in many zero values when the model fails to predict minority classes correctly.

The AUC metric, on the other hand, provides a more reliable measure of the model's performance, as it considers both true positive rate (sensitivity) and false positive rate (1-specificity) across various decision thresholds. By using the AUC metric, we can better understand the trade-offs between sensitivity and specificity, making it a more suitable evaluation metric for our imbalanced dataset. This choice allows us to effectively compare and assess the performance of different models and techniques employed in our project.

- Y-a-t-il un scheduler ?
Yes, in the code there a Learning rate reduction and early stopping.

Learning rate reduction: decay learning rate if no val loss improvement in this epoch (Patience = 1 epoch)

Early stopping : break if no val loss improvement in 3 epochs (Patience = 3 epoch)

Maybe we can increase the patience value to get better performance, because training a model have randomness.

- Charger le dataset, le vérifier (ouvrir quelques images) Faire un histogram par class (avec Excel ou Libre office) pour validation et training.
Explore_chexphotodataset.ipynb and Explore_nihdataset.ipynb


# Here is the original readme
# reproduce-chexnet
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/jrzech/reproduce-chexnet/master?filepath=Explore_Predictions.ipynb)

Provides Python code to reproduce model training, predictions, and heatmaps from the [CheXNet paper](https://arxiv.org/pdf/1711.05225) that predicted 14 common diagnoses using convolutional neural networks in over 100,000 NIH chest x-rays.

![Illustration](illustration.png?raw=true "Illustration")


## Getting Started:
Click on the `launch binder` button at the top of this `README` to launch a remote instance in your browser using [binder](https://mybinder.org/). This requires no local configuration, but it can take a couple minutes to launch. Open `Explore_Predictions.ipynb`, run all cells, and follow the instructions provided to review a selection of included [chest x-rays from NIH](https://arxiv.org/pdf/1705.02315.pdf).

To configure your own local instance (assumes [Anaconda is installed](https://www.anaconda.com/download/); can be run on paperspace GPU instance if you do not have a GPU):

```git clone https://www.github.com/jrzech/reproduce-chexnet.git
cd reproduce-chexnet
conda env create -f environment.yml
source activate reproduce-chexnet
python -m ipykernel install --user --name reproduce-chexnet --display-name "Python (reproduce-chexnet)"
```
## Updates (2022)

Changes in libraries available through conda channels and non-compability of older versions of pytorch / torchvision with newer CUDA drivers caused the original build to break.  

This library was updated in late 2022 so anyone interested could continue to use it. (1) environment.yml was updated and slight edits to code were made to ensure compatibility with newer version of pytorch/torchvision (2) NIH CXR labels were updated to latest version. Given changes in torchvision model naming conventions and updated labels, a new model was retrained; reported AUC numbers are based on this retrained model. 

Given [expected variability in predictions of retrained deep learning models](https://arxiv.org/pdf/1912.03606.pdf), predictions vary from the model originally posted in 2018. The original 2018 model and predictions are shared in a /pretrained-old folder, but will require you to create a compatible environment with the older pytorch 0.4.0 and torchvision 0.2.0 to use them interactively.   

## Replicated results:
This reproduction achieved diagnosis-level AUC as given below compared to original paper:

<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>retrained auc</th>
      <th>chexnet auc</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Atelectasis</th>
      <td>0.8180</td>
      <td>0.8094</td>
    </tr>
    <tr>
      <th>Cardiomegaly</th>
      <td>0.9090</td>
      <td>0.9248</td>
    </tr>
    <tr>
      <th>Consolidation</th>
      <td>0.8002</td>
      <td>0.7901</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>0.8945</td>
      <td>0.8878</td>
    </tr>
    <tr>
      <th>Effusion</th>
      <td>0.8827</td>
      <td>0.8638</td>
    </tr>
    <tr>
      <th>Emphysema</th>
      <td>0.9316</td>
      <td>0.9371</td>
    </tr>
    <tr>
      <th>Fibrosis</th>
      <td>0.8251</td>
      <td>0.8047</td>
    </tr>
    <tr>
      <th>Hernia</th>
      <td>0.9175</td>
      <td>0.9164</td>
    </tr>
    <tr>
      <th>Infiltration</th>
      <td>0.7156</td>
      <td>0.7345</td>
    </tr>
    <tr>
      <th>Mass</th>
      <td>0.8377</td>
      <td>0.8676</td>
    </tr>
    <tr>
      <th>Nodule</th>
      <td>0.7756</td>
      <td>0.7802</td>
    </tr>
    <tr>
      <th>Pleural_Thickening</th>
      <td>0.7889</td>
      <td>0.8062</td>
    </tr>
    <tr>
      <th>Pneumonia</th>
      <td>0.7617</td>
      <td>0.7680</td>
    </tr>
    <tr>
      <th>Pneumothorax</th>
      <td>0.8776</td>
      <td>0.8887</td>
    </tr>
  </tbody>
</table>
</div>

## Results available in pretrained folder:
- `aucs.csv`: test AUCs of retrained model vs original ChexNet reported results
- `checkpoint`: saved model checkpoint
- `log_train`: log of train and val loss by epoch
- `preds.csv`: individual probabilities for each finding in each test set image predicted by retrained model

## NIH Dataset
To explore the full dataset, [download images from NIH (large, ~40gb compressed)](https://nihcc.app.box.com/v/ChestXray-NIHCC),
extract all `tar.gz` files to a single folder, and provide path as needed in code. You can use batch download script provided by NIH researchers included in this repo:

```
python nih_batch_download_zips.py
```

## Train your own model!
Please note: a GPU is required to train the model. You will encounter errors if you do not have a GPU available and compatible CUDA installed and you attempt to retrain. With a GPU, you can retrain the model with `retrain.py`. Make sure you download the full NIH dataset before trying this. If you run out of GPU memory, reduce `BATCH_SIZE` from its default setting of 16.

Please ensure your CUDA driver is compatible with the CUDA toolkit (v11.3) installed by default.

If you do not have a GPU, but wish to retrain the model yourself to verify performance, you can replicate the model with paperspace, Amazon EC2, Google Colaboratory, or other online cloud GPU services. If you're starting from scratch, [paperspace](http://www.paperspace.com) is easy to get started with. 

## Note on training
I use SGD+momentum rather than the Adam optimizer as described in the original [paper](https://arxiv.org/pdf/1711.05225.pdf). I achieved better results with SGD+momentum, as has been reported in [other work](https://arxiv.org/pdf/1705.08292.pdf).

## Note on data
A sample of 621 test NIH chest x-rays enriched for positive pathology is included with the repo to faciliate immediate use and exploration in the `Explore Predictions.ipynb` notebook. The [full NIH dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) is required for model retraining.

## Use and citation
My goal in releasing this code is to increase transparency and replicability of deep learning models in radiology. I encourage you to use this code to start your own projects. If you do, please cite the repo:

```@misc{Zech2018,
  author = {Zech, J.},
  title = {reproduce-chexnet},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jrzech/reproduce-chexnet}}
}
```

## Acknowledgements
With deep gratitude to researchers and developers at PyTorch, NIH, Stanford, and Project Jupyter, on whose generous work this project relies. With special thanks to Sasank Chilamkurthy, whose demonstration code was incorporated into this project. PyTorch is an incredible contribution to the research community.
