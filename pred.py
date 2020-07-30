import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *

def pred_and_evaluate(data,model, evaluateL2, evaluateL1):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    while(1):
        X,Y=data.get_data()
        #print(data.get_pos())
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


def pred(data,model,output):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    #test = None

    while(1):
        X,Y=data.get_data()
        #print(data.get_pos())
        if data.get_pos()>=data.P:
            X = torch.unsqueeze(X,dim=1)
            X = X.transpose(2,3)
            with torch.no_grad():
                output = model(X)
            output = torch.squeeze(output)
            if len(output.shape)==1:
                output = output.unsqueeze(dim=0)
        else:
            output=Variable(torch.zeros(1,data.m).to(data.device))
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        
        
        if not data.go_next():
            break

   

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy().reshape((data.n,data.m))
    print(predict.shape)
    if data.normalize==1:
        predict=np.multiply(predict,data.max)
        Ytest=np.multiply(Ytest,data.max)
    elif data.normalize==2:
        scale=data.scale.data.cpu().numpy()
        predict=np.multiply(predict,scale)
        Ytest=np.multiply(Ytest,scale)
    '''
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            p=predict[i][j]
            y=Ytest[i][j]
            print(p)
            print(y)
            print("")
    '''
    print(predict.shape)
    predict.tofile(args.output)
    

parser = argparse.ArgumentParser(description='predictor')
parser.add_argument('--data', type=str, default='./data/solar_AL_test.dat',
                    help='location of the data file')

parser.add_argument('--model', type=str, default='model/model.pt',
                    help='path to the model')
parser.add_argument('--output', type=str, 
                    help='path to the predicted ')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--n', type=int, default=10512,help='num of time steps')
parser.add_argument('--m', type=int, default=137,help='num of variables ')
parser.add_argument('--device',type=str,default='cuda:1',help='')



parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
#parser.add_argument('--horizon', type=int, default=3)






args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

Data = DataLoaderS_pred(args.data, args.n, args.m, device, args.seq_in_len, args.normalize)
#evaluateL2 = nn.MSELoss(size_average=False).to(device)
#evaluateL1 = nn.L1Loss(size_average=False).to(device)
with open(args.model, 'rb') as f:
    model = torch.load(f)

pred(Data, model,args.output)
                                         
#print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
