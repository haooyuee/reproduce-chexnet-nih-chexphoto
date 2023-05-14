import cxr_dataset as CXR
import eval_model as E
import model as M
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default="test1")
    parser.add_argument('--dataset_path', type=str, default="D:/googleDrive/ciusss/reproduce-chexnet-master/imgdata/images-NIH-224")
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='')

    cfg = parser.parse_args()
    print(cfg)
    # you will need to customize PATH_TO_IMAGES to where you have uncompressed
    # NIH images
    #PATH_TO_IMAGES = "D:/googleDrive/ciusss/reproduce-chexnet-master/imgdata/images-NIH-224"
    #WEIGHT_DECAY = 1e-4
    #LEARNING_RATE = 0.01
    preds, aucs = M.train_cnn(cfg)

