import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import math
import os
import pickle
import numpy as np
from Utils import model_statistics


class residual_block(nn.Module):
	def __init__(self, inc=4, outc=4, ks=(3,3,3), pads=(1,1,1)):
		super(residual_block, self).__init__()

		self.conv1 = nn.Conv3d(in_channels=inc, out_channels=outc, kernel_size=ks, padding=pads)
		self.bn1 = nn.BatchNorm3d(outc)
		self.avt1 =  nn.ELU()
		
		self.conv2 = nn.Conv3d(in_channels=inc, out_channels=outc, kernel_size=ks, padding=pads)
		self.bn2 = nn.BatchNorm3d(outc)
		self.avt2 =  nn.ELU()		

	def forward(self, x): # x shape: [bs, channel, x, y, z]
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.avt1(out)
		#print(out.shape)
		out = self.conv2(x)
		out = self.bn2(out)
		out = self.avt2(out)

		#print (out.shape)
		return x+out		

class transformer_encoder(nn.Module):
	def __init__(self, args):
		super(transformer_encoder, self).__init__()
	
		encoder_layer = nn.TransformerEncoderLayer(d_model=args.nhidden, nhead=8, dim_feedforward=args.fold_encoder_feedforward)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

	def forward(self, x):
		# x shape: [5*5*5, batch_size, nhidden]
		output = self.transformer_encoder(x)
		return output

class PositionalEncoding_3D(nn.Module):

	def __init__(self, nhidden, dropout=0.1, max_len=10):
		super(PositionalEncoding_3D, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, max_len, max_len, nhidden)

		pe_1d = torch.zeros(max_len, nhidden)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, nhidden, 2).float() * (-math.log(10000.0) / nhidden))
		# print (position.shape, div_term.shape, (position*div_term).shape)


		pe_1d[:, 0::2] = torch.sin(position * div_term)
		pe_1d[:, 1::2] = torch.cos(position * div_term)

		#print (pe_1d.shape, pe_1d.unsqueeze(1).unsqueeze(1).shape)

		pe += pe_1d.unsqueeze(0).unsqueeze(0).repeat(max_len, max_len, 1, 1)
		pe += pe_1d.unsqueeze(1).unsqueeze(0).repeat(max_len, 1, max_len, 1)
		pe += pe_1d.unsqueeze(1).unsqueeze(1).repeat(1, max_len, max_len, 1)

		# print (pe_1d.shape, pe_1d.unsqueeze(0).unsqueeze(0).repeat(max_len, max_len, 1, 1).shape, \
		# 	pe_1d.unsqueeze(1).unsqueeze(0).repeat(max_len, 1, max_len, 1).shape, \
		# 	pe_1d.unsqueeze(1).unsqueeze(1).repeat(1, max_len, max_len, 1).shape, pe.shape)
		
		#pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)
		# pe [5, 5, 5, 64]

	def forward(self, x):
		# x [bsz, 5, 5, 5, 64]
		x = x + self.pe[:x.size(1), :x.size(2), :x.size(3), :]
		return self.dropout(x)


class fold_encoder(nn.Module):
	def __init__(self, args):
		super(fold_encoder, self).__init__()

		#  convolutional block
		self.rb_4 = nn.ModuleList([residual_block(inc=4, outc=4) for i in range(2)])
		self.downsmp1 = nn.Conv3d(in_channels=4, out_channels=16, kernel_size=(3,3,3), stride = (2,2,2), padding=(1,1,1))
		self.bn1 = nn.BatchNorm3d(16)
		self.avt1 =  nn.ELU()

		self.rb_8 = nn.ModuleList([residual_block(inc=16, outc=16) for i in range(2)])
		self.downsmp2 = nn.Conv3d(in_channels=16, out_channels=args.nhidden, kernel_size=(3,3,3), stride = (2,2,2), padding=(1,1,1))
		self.bn2 = nn.BatchNorm3d(args.nhidden)
		self.avt2 =  nn.ELU()

		self.rb_16 = nn.ModuleList([residual_block(inc=args.nhidden, outc=args.nhidden) for i in range(2)])

		self.pos_encoder_3d = PositionalEncoding_3D(args.nhidden, args.dropout)
		self.transformer_encoder = transformer_encoder(args)
		#self.linear = nn.Linear(16, 64)

		# transformer block

	def forward(self, x):# x shpae: [batch_size, 20, 20, 20, 4]
		x = torch.transpose(x, 1, -1)
		out = x     

		#print (out.shape)
		for i in range(2):
			out = self.rb_4[i](out)

		#print("before downsampling: ", out.shape)
		out = self.avt1(self.bn1(self.downsmp1(out)))

		#print ("after downsampling: ", out.shape)
		for i in range(2):
			out = self.rb_8[i](out)

		#print("before downsampling: ", out.shape)
		out = self.avt2(self.bn2(self.downsmp2(out)))
		#print ("after downsampling: ", out.shape)

		for i in range(2):
			out = self.rb_16[i](out)

		out = torch.transpose(out, -1, 1)
		#print ("before positional encoding: ", out.shape) # [bsz, 5, 5, 5, 64]

		out = self.pos_encoder_3d(out)

		out = torch.flatten(out, start_dim=1, end_dim=3)
		out = torch.transpose(out, 0, 1)
		out = self.transformer_encoder(out)
		#print ("fold encoder out shape:", out.shape)
		# The output shape should be [125, batch_size, 64]
		return out


class fold_classification(nn.Module):
	def __init__(self, args, hs=432, share_hs=960, nfolds=1227):
		super(fold_classification, self).__init__()
		self.fold_encoder = fold_encoder(args)
		self.l1 = nn.Linear(args.nhidden*15, nfolds)
		self.flat = nn.Flatten()
		self.avgpool = nn.AvgPool1d(8)
	def forward(self, x):
		out = self.fold_encoder(x) # x [125, bsz, 64]
		out = out.transpose(0, 1).transpose(1,2)  # out[bsz, 64, 125]
		print (out.shape)
		out = self.avgpool(out)
		print (out.shape)
		out = self.flat(out)
		print(out.shape)
		out = self.l1(out)

		return out


class fold_dataset(Dataset):
	def __init__(self, path="../cath_data/domain_dict.pkl", mode='train'):
		self.mode = mode
		with open(path, "rb") as f:
			domain_dict = pickle.load(f)

		fold_index= np.loadtxt("../cath_data/fold_index.txt", dtype='str')
		self.name_list=[]
		self.label=[]
		for i in domain_dict:
			if domain_dict[i]['train']==True and mode=='train':
				self.name_list.append(i.replace('/','-'))
				self.label.append( fold_index.tolist().index(domain_dict[i]['fold']) )

			elif domain_dict[i]['train']==False and mode=='test':
				self.name_list.append(i.replace('/','-'))
				self.label.append( fold_index.tolist().index(domain_dict[i]['fold']) )

	def __len__(self):
		return len(self.name_list)

	def __getitem__(self, idx):
		x = np.load("../cath_data/fold_features/"+self.name_list[idx]+".npy")
		y = self.label[idx]
		return {'x':x, 'y': y}

def data_split():

	# split training and test data:
	with open("../cath_data/domain_dict.pkl", "rb") as f:
		domain_data = pickle.load(f)

	fold_dict={}
	for i in domain_data:
		if domain_data[i]['fold'] not in fold_dict:
			fold_dict[domain_data[i]['fold']] = [i]
		else:
			fold_dict[domain_data[i]['fold']].append(i)
	
	fold_train=[]
	fold_test = []
	for i in fold_dict:
		ind_list=[]
		for j in fold_dict[i]:
			ind_list.append(j)
		ind_list=np.array(ind_list)
		np.random.shuffle(ind_list)
		for i in range(len(ind_list)):
			if i<0.9*float(len(ind_list)):
				domain_data[ind_list[i]]['train']=True
				fold_train.append(ind_list[i])
			else:
				domain_data[ind_list[i]]['train']=False
				fold_test.append(ind_list[i])

	np.savetxt("../cath_data/fold_train.txt",fold_train, fmt='%s')
	np.savetxt("../cath_data/fold_test.txt",fold_test,  fmt='%s')
	with open("../cath_data/domain_dict.pkl", "wb") as f:
		pickle.dump(domain_data, f)

def train_epoch(model, Adam_opt, dataLoader, e):
    
	#bsz=args.batch_size
	model.train()
	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
		device = 'cuda'
	else:
		device = 'cpu'


	model.to(device)
	model = model.double()

	for bt_ind, smp  in enumerate(dataLoader):

		x = smp['x'].to(device).double()
		y = smp['y'].to(device)
		Adam_opt.zero_grad()
		preds = model(x)

		loss =nn.CrossEntropyLoss()(preds, y)
		loss.backward()
		Adam_opt.step()

		with open(args.model_save+".train.txt", "a") as f:
		    f.write("epochs:%d batch %d/%d loss: %.4f\n" %(e, bt_ind, len(dataLoader), loss))
		print ("epochs:%d batch %d/%d loss: %.4f" %(e, bt_ind, len(dataLoader), loss))
        
def eval(model, dataLoader, e, numofsamples):
	model.eval()

	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
		device = 'cuda'
	else:
		device = 'cpu'

	model.to(device)
	model = model.double()
	top1=0.
	top3=0.
	top5=0.
	top10=0.

	for bt_ind, smp  in enumerate(dataLoader):

		x = smp['x'].to(device).double()
		y = smp['y'].to(device)
		
		preds = model(x)
		am = model_statistics.multi_class_accuracy(y, preds)

		top1+=am[0]
		top3+=am[1]
		top5+=am[2]
		top10+=am[3]

		print ("test batch %d/%d" %(bt_ind, len(dataLoader)))


	print ("overall performance: top1 %.3f\n top3 %.3f\n top5 %.3f\n top10 %3f\n" %(top1/numofsamples \
	, top3/numofsamples, top5/numofsamples, top10/numofsamples))



	with open(args.model_save+".eval.txt", "a") as f:
		f.write("epoch %d overall performance: top1 %.3f  top3 %.3f  top5 %.3f  top10 %3f\n" %(e, top1/numofsamples \
        , top3/numofsamples, top5/numofsamples, top10/numofsamples))


	return top1/numofsamples

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments for pretrain_seq.py')
	parser.add_argument('--nhidden', default=128, type=int)
	parser.add_argument('--dropout', default=0.1, type=float)
	parser.add_argument('--nfolds', default=1227, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--lr', default=1E-3, type=float)
	parser.add_argument('--fold_encoder_feedforward', default=512,  type=int)
	parser.add_argument('--model_save', default="pretrained_models/foldmodel1", type=str)
	parser.add_argument('--pretrained_model', default=None, type=str)
	parser.add_argument('--wd', default=0., type=float)
	args = parser.parse_args()


	rb1 = fold_encoder(args)

	num_params=0
	for i in rb1.state_dict():
		j=1
		for k in rb1.state_dict()[i].shape:
			j*=k
		num_params+=j

	print ("#num_params:", num_params)

	trainset = fold_dataset(mode='train')
	testset = fold_dataset(mode='test')
	trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
	testset_loader = DataLoader(testset,  batch_size=args.batch_size, shuffle=False, num_workers=8)

	model = fold_classification(args, nfolds=args.nfolds)
	Adam_opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
	
	best_acc=0.
	for e in range(1, args.epochs+1):
		train_epoch(model, Adam_opt, trainset_loader, e)
		acc = eval(model, testset_loader, e, len(testset))


		if acc > best_acc:
			best_acc=acc
			torch.save({
				'epoch': e,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': Adam_opt.state_dict(),
				'args': args
				}, args.model_save)

