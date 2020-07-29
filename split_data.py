import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--portion',type=float,default=0.8,help='dropout rate')
parser.add_argument('--data',type=str,default='data/solar_AL.txt',help='data path')
parser.add_argument('--output',type=str,help='output path')
args = parser.parse_args()



data=np.loadtxt(args.data,delimiter=',')
data=data.astype(np.float32)
n=data.shape[0]
start=int(args.portion*n)

data[start:].tofile(args.output)