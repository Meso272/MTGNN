import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *

def pred(data,model, evaluateL2, evaluateL1):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    while(1):
        X,Y=data.get_data()
        print(data.get_pos())
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)
        if not data.go_next():
            break

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation


parser = argparse.ArgumentParser(description='predictor')
parser.add_argument('--data', type=str, default='./data/solar_AL.txt',
                    help='location of the data file')

parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:1',help='')



parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
#parser.add_argument('--horizon', type=int, default=3)






args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

Data = DataLoaderS_pred(args.data, 0.6, 0.2, device, args.seq_in_len, args.normalize)
evaluateL2 = nn.MSELoss(size_average=False).to(device)
evaluateL1 = nn.L1Loss(size_average=False).to(device)
with open(args.save, 'rb') as f:
    model = torch.load(f)

test_acc, test_rae, test_corr = evaluate(Data, model, evaluateL2, evaluateL1,
                                         )
print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
