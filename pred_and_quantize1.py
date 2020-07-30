import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *


def quantize(data,pred,errorbound):
    radius=32768
    diff = data - pred
    quant_index = (int) (diff / error_bound) + 1
    if (quant_index < radius * 2) :
        quant_index =quant_index>> 1
        half_index = quant_index
        quant_index =quant_index<< 1
        quant_index_shifted=0
        if (diff < 0) :
            quant_index = -quant_index
            quant_index_shifted = radius - half_index
        else :
            quant_index_shifted = radius + half_index;
        
        decompressed_data = pred + quant_index * error_bound
        if abs(decompressed_data - data) > error_bound :
            return 0
        else:
            data = decompressed_data;
            return quant_index_shifted;
        
    else:
        return 0
    



def pred_and_quantize(data,model,errorbound,output_quantized,output_unpred):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    #test = None
    quantarray=[]
    unpred=[]
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
            output=output.data.cpu().numpy()[0]

        else:
            output=np.zeros(data.m)
        Y=Y.data.cpu().numpy()
       

        #scale = data.scale.expand(output.size(0), data.m)
        if data.normalize==2:
            scale=data.scale.data.cpu().numpy()
        for i in range(data.m):
            elex=output[i]
            eley=Y[i]
            if data.normalize==1:
                elex=elex*data.max
            elif data.normalize==2:
                elex=elex*scale[i]
            quantres=quantize(eley,elex,errorbound)
            quantarray.append(quantres)
            if quantres==0:
                unpred.append(eley)
        if not data.go_next():
            break


    qarray=np.array(quantarray,dtype=np.int16)
    uarray=np.array(unpred,dtype=np.float32)

    qarray.tofile(output_quantized)
    uarray.tofile(output_unpred)

   

    #predict = predict.data.cpu().numpy()
    #print(predict.shape)
    


    
    #print(predict.shape)
    #predict.tofile(args.output)
    

parser = argparse.ArgumentParser(description='predictor')
parser.add_argument('--data', type=str, default='./data/solar_AL_test.dat',
                    help='location of the data file')

parser.add_argument('--model', type=str, default='model/model.pt',
                    help='path to the model')
parser.add_argument('--outq', type=str, 
                    help='quantized path to the predicted ')
parser.add_argument('--outu', type=str, 
                    help='unpred path to the predicted ')
parser.add_argument('--error', type=float, default=1e-3)
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

pred_and_quantize(Data, model,args.error,args.outq,args.outu)
                                         
#print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
