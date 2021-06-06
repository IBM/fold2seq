import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import OrderedDict
import argparse
import numpy as np
import math
import os
import pickle
import numpy as np
import random
from Utils import amino_acid
from Utils import model_statistics
import train
import time

def top_k_sampling(token_preds, k=5):
	token_preds = torch.topk(token_preds, k=k, dim=-1)
	tokens_dis = torch.distributions.Categorical(token_preds.values)
	smp = tokens_dis.sample()
	token_smp = token_preds.indices[torch.arange(0, smp.size(0)) , smp]

	return token_smp

def greedy_sampling(token_preds):
        return torch.max(token_preds, 1).indices

class fold_dataset(Dataset):
	def __init__(self, args):


		self.name_list=[]
		self.args=args

		with open(args.data_path, "rb") as f:
			domain = pickle.load(f)
		

		for i in domain:
			for j in range(args.n):
				self.name_list.append(i.replace('/','-'))
		print ("mode=", args.mode, " ", len(self.name_list))

	def __len__(self):
		return len(self.name_list)

	def __getitem__(self, idx):
		x = np.load(self.args.fold_path+self.name_list[idx]+".npy")
		return [torch.tensor(x).float(), self.name_list[idx]]

def inference(model, args, fold):
	# fold should be [bsz, 20, 20, 20, 4]
	model.eval()

	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
		device = 'cuda'
	else:
		device = 'cpu'
	#device='cpu'
	model.to(device)
	fold=fold.to(device)
	#tape_token_encode=[2] # 2 is the starting encode in tape

	seq_embed=torch.tensor([[21] for i in range(fold.size(0))]).to(device)
	start_time = time.time()

	t1=time.time() 
	with torch.no_grad():
		for i in range(args.maxlen):
			#tape_out = tape_model(torch.tensor([tape_token_encode]).to(device))
			print (seq_embed.shape, fold.shape)
			c1, c2, c3, c4, c5,c6,c7 = model(seq_embed, fold, mode='inference')

			c1=c1.transpose(0,1)
			print ("c1 shape", c1.shape)
			# c1 shape : [seq_len, bsz, 21]
			c1 = torch.softmax(c1, dim=-1)
			if len(args.decodetype)<=2:
				tokens = top_k_sampling(c1[-1], int(args.decodetype)) # tokens should have shape [bsz, 1]
			elif args.decodetype=='greedy':
				tokens = greedy_sampling(c1[-1])
			seq_embed = torch.cat((seq_embed, tokens.unsqueeze(1)), 1)
			print ( seq_embed.shape)
			t3=time.time()
			with open("runtime_fold2seq.txt_cpu", "a") as fout:
				fout.write ("%d %.4f\n" %(i+1, (t3-t1)/args.batch_size))
	end_time = time.time()
	print ("batch time", end_time-start_time)
	return seq_embed


def main():
	parser = argparse.ArgumentParser(description='Arguments for inference.py')
	parser.add_argument('--data_path', default="./domain_dict.pkl", type=str)
	parser.add_argument('--fold_path', default="../data/fold_features/", type=str)
	parser.add_argument('--trained_model', default=None, type=str)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--n', default=100, type=int)
	parser.add_argument('--output', default='./gen_seq.txt', type=str)
	parser.add_argument('--mode', default='test2', type=str)
	parser.add_argument('--maxlen', default=200, type=int)
	parser.add_argument('--decodetype', default="5", type=str)
	parser.add_argument('--lba0', default=1, type=float)  # coefficient before  fold2seq_loss
	parser.add_argument('--lba1', default=0, type=float)  # coefficient before seq2seq_loss
	parser.add_argument('--lba2', default=0, type=float)  # coefficient before foldclass_loss
	parser.add_argument('--lba3', default=0, type=float)  # coefficient before  seqclass_loss
	parser.add_argument('--lba4', default=0, type=float)  # coefficient before  sim_loss
	parser.add_argument('--lba5', default=0, type=float)
	parser.add_argument('--lba6', default=0, type=float)

	args = parser.parse_args()

	if args.trained_model == None:
		raise ValueError("Must specify a trained model to be inferenced")
	
	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
		device = 'cuda'
	else:
		device = 'cpu'

	device='cpu'
	trained_dict = torch.load(args.trained_model, map_location = 'cpu')
	trained_dict['args'].device = device
	trained_dict['args'].lba0=1
	trained_dict['args'].lba1=0
	trained_dict['args'].lba2=0
	trained_dict['args'].lba3=0
	trained_dict['args'].lba4=0
	trained_dict['args'].lba5=0
	trained_dict['args'].lba6=0
	print (trained_dict['args'], trained_dict['epoch'], trained_dict['metric'])
	
	model_name = args.trained_model.split('/')[1]
	print ("output path: ", args.output)

	model = train.fold_classification_generator(trained_dict['args'])

	new_state_dict = OrderedDict()
	for k,v in trained_dict['model_state_dict'].items():
		if 'module' ==k[:6]: # that means the pretrained model having dataparllel
			new_state_dict[k[7:]]=v
		else:
			new_state_dict[k]=v

	model.load_state_dict(new_state_dict, strict=False)

	if torch.cuda.device_count() > 1:
		print("Let's use ",torch.cuda.device_count()," GPUs!")
		model = nn.DataParallel(model)
		args.batch_size*=torch.cuda.device_count()

	testset = fold_dataset(args)
	testset_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

	for bt_ind, fold  in enumerate(testset_loader):
		gen_seq_tokens = inference(model, args, fold[0])
		t = gen_seq_tokens.detach().cpu().numpy()
	
		
		with open(args.output, "a") as f:
			for i in range(len(fold[1])):
				s=''
				for j in range(args.maxlen):
					if t[i][j]==0:
						break
					s+=amino_acid.my_seqlabel[t[i][j]]
				print (s)
				f.write(fold[1][i]+" "+s+"\n")


if __name__=='__main__':
	main()
