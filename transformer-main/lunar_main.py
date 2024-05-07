import numpy as np
import time
import utils
import variables as var
from sklearn.metrics import roc_auc_score 
import LUNAR
import argparse

from transformer import Encoder
import torch

def main(args):
    avg_score = 0
    count = 0
    avg_RUNtime = 0
    for seed in [0,1,2,3,4]:

        print("Running trial with random seed = %d" %seed)
        #load dataset (without negative samples)
        train_x, train_y, val_x, val_y, test_x, test_y = utils.load_dataset(args.dataset,seed)

        start = time.time()
        test_out = LUNAR.run(train_x,train_y,val_x,val_y,test_x,test_y,args.dataset,seed,args.k,args.samples,args.train_new_model)
        end = time.time()

        score = 100*roc_auc_score(test_y, test_out)

        print('Dataset: %s \t Samples: %s \t k: %d \t Score: %.4f \t Runtime: %.2f seconds' %(args.dataset,args.samples,args.k,score,(end-start)))

# 求平均Score和平均Runtime###########
        avg_score = avg_score + score
        RUNtime = end-start
        avg_RUNtime = avg_RUNtime + RUNtime
        count = count+1
    avg_score = avg_score/count
    avg_RUNtime = avg_RUNtime/count
    print('平均 Score: %.4f \t Runtime: %.4f' %(avg_score,avg_RUNtime))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'NYC')
    parser.add_argument("--samples", type = str, default = 'MIXED', help = 'Type of negative samples for training')
    parser.add_argument("--k", type = int, default =100)
    parser.add_argument("--train_new_model", action="store_true",default=True, help = 'Train a new model vs. load existing model')
    args = parser.parse_args()

    main(args)
    print(args.dataset)
    print(args.k)
    print(args.train_new_model)
