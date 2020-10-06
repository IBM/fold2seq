import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse
from collections import OrderedDict
import numpy as np
import math
import os
import pickle
import numpy as np
import random
from Utils import amino_acid
from Utils import model_statistics
from generator import fold_classification_generator

class generator_dataset(Dataset):
	def __init__(self, args, domain_data, mode):
		
		self.args=args
		self.domain_data = {}
		#fold_index= np.loadtxt("../cath_data/fold_index.txt", dtype='str')
		self.name_list=[]
		self.permute=[]
		for i in domain_data:
				if domain_data[i]['mode']==mode and len(domain_data[i]['seq'])<=args.maxlen:
					self.domain_data[i]=domain_data[i]
					self.name_list.append(i)
					self.permute.append((0,1,2,3))
		
		if mode=='train' and args.augmentation==1:
			self.permute=[(0,1,2,3) for i in range(len(self.name_list))] + [(0,2,1,3) for i in range(len(self.name_list))] + [(1,0,2,3) for i in range(len(self.name_list))] \
			+ [(1,2,0,3) for i in range(len(self.name_list))] + [(2,0,1,3) for i in range(len(self.name_list))] + [(2,1,0,3) for i in range(len(self.name_list))]
			self.name_list = self.name_list+self.name_list+self.name_list+self.name_list+self.name_list+self.name_list			
		print (mode+" dataset: "+str(len(self.name_list)))


	def __len__(self):
		return len(self.name_list)

	def __getitem__(self, idx):
		fold_feat = np.load("../cath_data/fold_features/"+self.name_list[idx].replace('/','-')+".npy")
		seq_feat = self.domain_data[self.name_list[idx]]['embed'][:self.args.maxlen+2]
		classlabel  = self.domain_data[self.name_list[idx]]['fold_index']
		seq_padding =self.domain_data[self.name_list[idx]]['padding'][:self.args.maxlen+2]
		return {'fold_feat':torch.tensor(fold_feat).float().permute(self.permute[idx]), 'seq_feat': torch.tensor(seq_feat).long(), "classlabel": torch.tensor(classlabel), 'pad':torch.tensor(seq_padding)}

def var_len_data_pre(domain_data,  batch_size, mode='train'):
    #  Batch those sequences that have simlar length. We pad sequences to the maximal length in every 512 sequences
    #  ordered from short to long. 



    fold_index = np.loadtxt("../cath_data/fold_index.txt", dtype=str)

    keys=[]
    for i in domain_data:
        if domain_data[i]['mode'] == mode:
            keys.append(i)

        domain_data[i]['fold_index']  = fold_index.tolist().index(domain_data[i]['fold'])
        domain_data[i]['decoderlabel'] = amino_acid.seqlabel(domain_data[i]['seq'])

    sorted_keys = sorted(keys, key=lambda x: len(domain_data[x]['seq']))

    batch_ind_all = []
    bz=0
    clen = len(domain_data[sorted_keys[0]]['seq'])
    bsz_list=[]
    for i in range(len(sorted_keys)):
        key = sorted_keys[i]
        if len(domain_data[key]['seq']) == clen and bz<=batch_size:
            bsz_list.append(i)
            bz+=1
        else:
            batch_ind_all.append(bsz_list)
            bsz_list=[i]
            bz=1
            clen = len(domain_data[key]['seq'])

    batch_ind_all.append(bsz_list)

    print (mode, "#proteins:", len(sorted_keys), "#batches:", len(batch_ind_all))

    return sorted_keys, batch_ind_all

# -----sorted_keys:  The sorted list of keys
# -----batch_ind:    The index of batch 
# -----domain_data:  The domain data

def dataloader(args, domain_data, sorted_keys, bt, device):

    seq = []
    fold = []
    foldlabel=[]
    decoderlabel=[]
    for i in bt:
        keys = sorted_keys[i]  
        #print (len(domain_data[keys]['seq']))
        #seq.append(torch.load("../cath_data/seq_features_nopads/"+keys.replace('/','-')+".pt", map_location=device))
        #if args.encoder == 'fold' or args.encoder == 'both':
        seq_tokens = amino_acid.transformer_integer(domain_data[keys]['seq'])
        seq.append(seq_tokens)
        fold.append(np.load("../cath_data/fold_features/"+keys.replace('/','-')+".npy"))  
        foldlabel.append(domain_data[keys]['fold_index'])
        decoderlabel.append(seq_tokens[1:])

    return torch.tensor(seq).long(), torch.tensor(fold).float(), \
    torch.tensor(foldlabel), torch.tensor(decoderlabel)

def loss(args, preds, label, added_loss=[0,0,0,0,0,0], bsz=0):
	#  L = lba0*fold2seq_loss + lba1*seq2seq_loss + lba2*foldclass_loss + lba3*seqclass_loss 
	#  + lba4*sim_score
	L=torch.tensor(0.).to(args.device)
	#label['seqlabel'] = label['seqlabel'].transpose(0,1).flatten()  # labels shape [(seqs+1)*bsz]
	loss_list=[None, None, None, None, None, None]

	if args.lba0!=0:
		preds['fold2seq_preds'] = (preds['fold2seq_preds'].transpose(0,1).contiguous())[:-1].view(-1, args.ntokens-2) # preds shape: [(seqs+1)*bsz, 21] 
		fold2seq_loss =nn.CrossEntropyLoss(ignore_index=22)(preds['fold2seq_preds'], label['seqlabel'].long())
		L+=args.lba0*fold2seq_loss
		loss_list[0]=fold2seq_loss.detach().cpu().numpy()
		added_loss[0]+=loss_list[0]*bsz
		#print (loss_list, loss_list[0])

	if args.lba1!=0:
		preds['seq2seq_preds'] = (preds['seq2seq_preds'].transpose(0,1).contiguous())[:-1].view(-1, args.ntokens-2) # preds shape: [(seqs+1)*bsz, ntokens] 
		seq2seq_loss =nn.CrossEntropyLoss(ignore_index=22)(preds['seq2seq_preds'], label['seqlabel'].long())
		L+=args.lba1*seq2seq_loss
		loss_list[1]=seq2seq_loss.detach().cpu().numpy()
		added_loss[1]+=loss_list[1]*bsz

	if args.lba2!=0:
		foldclass_loss = nn.CrossEntropyLoss()(preds['foldclass_preds'], label['classlabel'].long() )
		L+=args.lba2*foldclass_loss
		loss_list[2]=foldclass_loss.detach().cpu().numpy()
		added_loss[2]+=loss_list[2]*bsz

	if args.lba3!=0:
		seqclass_loss = nn.CrossEntropyLoss()(preds['seqclass_preds'], label['classlabel'].long() )
		L+=args.lba3*seqclass_loss
		loss_list[3]=seqclass_loss.detach().cpu().numpy()
		added_loss[3]+=loss_list[3]*bsz

	if args.lba4!=0:
		l40 = torch.mean(preds['sim_score'][0])
		L+=args.lba4*(l40)
		loss_list[4]=l40.detach().cpu().numpy()
		added_loss[4]+=loss_list[4]*bsz
		l41 = torch.mean(preds['sim_score'][1])
		L+=args.lba4*(l41)
		loss_list[4]=(l41).detach().cpu().numpy()
		added_loss[4]+=loss_list[4]*bsz

	loss_list[5] = L.detach().cpu().numpy()
	added_loss[5]+=loss_list[5]*bsz
	return loss_list, L


def result_print(args, epoch, loss_list, batch=None, am=None, mode='train'):
	with open(args.model_save+"."+mode+".txt", "a") as f:
		f.write("epochs:%d " %(epoch))
		print("epochs:%d " %(epoch), end='')
		if batch!=None:
			f.write("batch %d/%d: " %(batch[0], batch[1]))
			print("batch %d/%d: " %(batch[0], batch[1]), end='')
		else:
			f.write("overall: ")
			print("overall: ", end='')
		for i in loss_list:
			if i==None:
				f.write("None ")
				print("None ", end='')
			else:
				f.write("%.4f " %(i))
				print("%.4f " %(i), end='')
		print('\n', end='')
		f.write('\n')

		if am!=None:
				f.write("epoch %d fold2class performance: top1 %.3f  top3 %.3f  top5 %.3f  top10 %3f\n" %(epoch, \
					am[0][0], am[0][1], am[0][2], am[0][3]))
				f.write("epoch %d seqclass performance: top1 %.3f  top3 %.3f  top5 %.3f  top10 %3f\n" %(epoch, \
					am[1][0], am[1][1], am[1][2], am[1][3]))



def eval(model,args, dataloader, e):

	model.eval()
	model.to(args.device)
	seqclass_acc=np.zeros(4, dtype=float)
	foldclass_acc=np.zeros(4, dtype=float)

	add_loss=np.zeros(6, dtype=float)

	n=0.
	for bt_ind, smp  in enumerate(dataloader):
			#seqs, fold, classlabel, seqlabel = dataloader(args, domain_data, sorted_keys, batch_ind[idx1], args.device)
			seqs = smp['seq_feat'].to(args.device)
			fold = smp['fold_feat'].to(args.device)
			classlabel = smp['classlabel'].to(args.device)
			seqpads = smp['pad'].to(args.device)
			seqlabel = seqs.transpose(0,1)[1:].flatten()
			#seqlabel = seqlabel.to(args.device)
			#fold = fold.to(args.dev
			print (seqs.shape, fold.shape, classlabel.shape, seqlabel.shape, seqpads.shape)
			c0,c1,c2,c3,c4,c5,c6 = model(seqs, fold, padding_masking=seqpads)  # c0 shape: [bsz, maxlen+1, ntokens-2]
			preds = {'fold2seq_preds': c0, 'seq2seq_preds':c1, 
            'foldclass_preds':c2, 'seqclass_preds':c3,'sim_score': [c4,c5,c6]}

			label={'classlabel':classlabel, 'seqlabel':seqlabel}

			loss_list,L = loss(args, preds, label, add_loss, fold.size(0))

			if args.lba2!=0:
				am = model_statistics.multi_class_accuracy(classlabel, preds['foldclass_preds'])
				foldclass_acc += am
			if args.lba3!=0:
				am = model_statistics.multi_class_accuracy(classlabel, preds['seqclass_preds'])
				seqclass_acc += am            	

			n+=fold.size(0)	

			print ("test batch %d/%d " %(bt_ind, len(dataloader)))
			print ("batch_loss:", loss_list)
	add_loss=(add_loss/n).tolist()
	seqclass_acc/=n
	foldclass_acc/=n
	if args.lba0==0:
		add_loss[0]=None
	if args.lba1==0:
		add_loss[1]=None
	if args.lba2==0:
		add_loss[2]=None
	if args.lba3==0:
		add_loss[3]=None
	if args.lba4==0:
		add_loss[4]=None
	result_print(args, e, add_loss, batch=None, am=[foldclass_acc, seqclass_acc], mode='eval')

	if args.lba0!=0:
		return add_loss[0]
	if args.lba1!=0:
		return add_loss[1]
	if args.lba2!=0:
		return -foldclass_acc[0]
	if args.lba3!=0:
		return -seqclass_acc[0]


def train_epoch(model, args,Adam_opt, scheduler,  dataloader, e):
#  L = lba0*fold2seq_loss + lba1*seq2seq_loss + lba2*foldclass_loss + lba3*seqclass_loss 
#  + lba4*sim_score

    model.train()
    model.to(args.device)

    #bt_ind=1
    #for idx1 in idx:
    for bt_ind, smp  in enumerate(dataloader):
        #seqs, fold, classlabel, seqlabel = dataloader(args, domain_data, sorted_keys, batch_ind[idx1], args.device)
        seqs = smp['seq_feat'].to(args.device)
        fold = smp['fold_feat'].to(args.device)
        classlabel = smp['classlabel'].to(args.device)
        seqlabel = seqs.transpose(0,1)[1:].flatten()
        seqpads = smp['pad'].to(args.device)
        #seqlabel = seqlabel.to(args.device)
        #fold = fold.to(args.device)
        print (seqs.shape, fold.shape, classlabel.shape, seqlabel.shape, seqpads.shape)

       	label={'classlabel':classlabel, 'seqlabel':seqlabel}

        c0,c1,c2,c3,c4,c5,c6 = model(seqs, fold, padding_masking=seqpads)  # c0 shape: [bsz, maxlen+1, ntokens-2]
        preds = {'fold2seq_preds': c0, 'seq2seq_preds':c1,'foldclass_preds':c2, 'seqclass_preds':c3,'sim_score': [c4,c5,c6]}


        print ('seq input shape:', seqs.shape,'fold input shape', fold.shape,'decoder label shape', seqlabel.shape)

        loss_list,L = loss(args, preds, label)

        Adam_opt.zero_grad() 
        L.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        Adam_opt.step()
        if args.lr==-1:
           scheduler.step()

        result_print(args, e, loss_list, batch=[bt_ind, len(dataloader)], am=None, mode='train')
        #bt_ind+=1

def var_len_shuffle(domain_data, sorted_keys, idx):
    
    cl = len(domain_data[sorted_keys[0]]['seq'])
    st=0
    for i in range(len(sorted_keys)):
        key = sorted_keys[i]
        if len(domain_data[key]['seq']) != cl:
            cl = len(domain_data[key]['seq'])
            c = sorted_keys[st:i]
            random.shuffle(c)
            sorted_keys[st:i] = c
            st = i

    np.random.shuffle(idx)

def load_pretrained_model(model, args):
	if args.pretrained_model !=None:
		model_pre = torch.load(args.pretrained_model, map_location='cpu')
		
		print ("loading model:", model_pre['args'], '\nepoch:', model_pre['epoch'], 'loss:', model_pre['metric'])
		new_state_dict = OrderedDict()
		for k,v in model_pre['model_state_dict'].items():
			if 'module' ==k[:6]: # that means the pretrained model having dataparllel
				new_state_dict[k[7:]]=v
			else:
				new_state_dict[k]=v
				

		print (model.decoder_out_linear.bias)
		model.load_state_dict(new_state_dict, strict=False)
		print (model.decoder_out_linear.bias)
		

		if args.freeze_seq_encoder == 1:  # freeze the sequence encoder
			for param in model.seq_encoder.parameters():
				param.requires_grad = False
			for param in model.seq_embedding.parameters():
				param.requires_grad = False
			for param in model.positional_embedding.parameters():
				param.requires_grad = False
			for param in model.fold_classification_linear.parameters():
				param.requires_grad = False
		if args.freeze_seq_decoder == 1:  # freeze the sequence decoder
			for param  in model.seq_decoder.parameters():
				param.requires_grad = False
			for param in model.seq_embedding.parameters():
				param.requires_grad = False
			for param in model.positional_embedding.parameters():
				param.requires_grad = False
			for param in model.decoder_out_linear.parameters():
				param.requires_grad = False	


def main():

     
    parser = argparse.ArgumentParser(description='Arguments for pretrain_seq.py')
    parser.add_argument('--ntokens', default=23, type=int)
    parser.add_argument('--nhidden', default=128, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    parser.add_argument('--decoder_feedforward', default=512,  type=int)
    parser.add_argument('--seq_encoder_feedforward', default=512,  type=int)
    parser.add_argument('--fold_encoder_feedforward', default=512,  type=int)
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    #parser.add_argument('--seq_embedding', default='ProteinSeqTransformer', type=str)
    parser.add_argument('--nfolds', default=1227, type=int)
    parser.add_argument('--data_path', default="../cath_data/domain_dict_full.pkl", type=str)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--encoder', default='both', type=str)
    parser.add_argument('--lr', default=1E-3, type=float)
    parser.add_argument('--model_save', default="trained_models/model1", type=str)
    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--freeze_seq_encoder', default=1, type=int)
    parser.add_argument('--freeze_seq_decoder', default=0, type=int)
    parser.add_argument('--augmentation', default=0, type=int)
    parser.add_argument('--lba0', default=1, type=float)  # coefficient before  fold2seq_loss
    parser.add_argument('--lba1', default=0, type=float)  # coefficient before seq2seq_loss
    parser.add_argument('--lba2', default=0, type=float)  # coefficient before foldclass_loss
    parser.add_argument('--lba3', default=0, type=float)  # coefficient before  seqclass_loss
    parser.add_argument('--lba4', default=0, type=float)  # coefficient before  sim_loss
    parser.add_argument('--lba5', default=0, type=float)
    parser.add_argument('--lba6', default=0, type=float)
    args = parser.parse_args()

    with open(args.data_path, "rb") as f:
        domain_data = pickle.load(f)

    if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
        device = 'cuda'
    else:
        device = 'cpu'
    args.device = device
    with open("check_device", "w") as f:
        f.write(device)
    # sorted_keys_trainset, batch_ind_trainset = var_len_data_pre(domain_data, args.batch_size, mode='train')
    # sorted_keys_testset, batch_ind_testset = var_len_data_pre(domain_data, args.batch_size, mode='test1')
    # sorted_keys_testset2, batch_ind_testset2 = var_len_data_pre(domain_data, args.batch_size, mode='test2')
    # train_idx = np.arange(0, len(batch_ind_trainset))
    # test_idx = np.arange(0, len(batch_ind_testset))
    # test_idx2 = np.arange(0, len(batch_ind_testset2))

    model = fold_classification_generator(args) 
    print ("model params:", model_statistics.net_param_num(model))
    md = model.state_dict()
    for i in md:
        print (i, md[i].shape)
    # loading pretrained model if possible
    #print (model.linear.weight[0])
    load_pretrained_model(model, args)
    
    if torch.cuda.device_count() > 1:
       print("Let's use ",torch.cuda.device_count()," GPUs!")
       with open("check_device", "w") as f:
         f.write("Let's use "+str(torch.cuda.device_count())+" GPUs!")
       model = nn.DataParallel(model)
       args.batch_size*=torch.cuda.device_count()

    trainset = generator_dataset(args, domain_data, mode='train')
    testset1 = generator_dataset(args, domain_data, mode='test1')
    testset2 = generator_dataset(args, domain_data, mode='test2')
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testset1_loader = DataLoader(testset1,  batch_size=args.batch_size, shuffle=False, num_workers=8)
    testset2_loader = DataLoader(testset2,  batch_size=args.batch_size, shuffle=False, num_workers=8)




    #print (model.linear.weight[0])

    #eval(model,args, sorted_keys_testset, batch_ind_testset, test_idx, domain_data, 0)

    if args.lr!=-1:
       Adam_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
       scheduler=None
    else:
       Adam_opt = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1E-09)
       lambdalr = lambda x: args.nhidden**(-0.5)*min((x+0.1)**(-0.5), x*((4000)**-1.5))
       scheduler = torch.optim.lr_scheduler.LambdaLR(Adam_opt, lr_lambda=lambdalr)
    best_m=eval(model,args, testset1_loader, 0)
    for e in range(1, args.epochs+1):

        #train_epoch(model,args, Adam_opt, sorted_keys_trainset, batch_ind_trainset, train_idx,domain_data, e)       
        #metric = eval(model,args, sorted_keys_testset, batch_ind_testset, test_idx, domain_data, e)
        #eval(model,args, sorted_keys_testset2, batch_ind_testset2, test_idx2, domain_data, e)
        if args.lba4!=0:
           args.lba4=1.0/2**(e-3.)
        train_epoch(model, args, Adam_opt, scheduler  ,trainset_loader,e )
        m1 = eval(model, args, testset1_loader, e)
        m2 = eval(model, args, testset2_loader, e)

        #if m2<2.9 and  m1 <= best_m:
            #best_m=m1
        torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': Adam_opt.state_dict(),
                'lrschedule_state_dict': scheduler.state_dict(),
                'args': args,
                'metric': [m1, m2]
        }, args.model_save+".e"+str(e))
  

if __name__=='__main__':
	main()
