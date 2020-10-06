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

# autoregressive sequence decoder
# current two versions:
#   1. transformer
#   2. LSTM
class transformer_decoder(nn.Module):
	def __init__(self, args):
		super(transformer_decoder, self).__init__()
	
		decoder_layer = nn.TransformerDecoderLayer(d_model=args.nhidden, nhead=8, dim_feedforward=args.decoder_feedforward)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
		self.args = args
		self.nhidden = args.nhidden
		#self.pos_encoder = PositionalEncoding(args.nhidden, args.dropout)
		#self.encoder = nn.Embedding(args.ntokens, args.nhidden)
		#self.init_weights()

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	# def init_weights(self):
	# 	initrange = 0.1
	# 	self.encoder.weight.data.uniform_(-initrange, initrange)


	def forward(self, seq, fold_embed, tgt_padding_masking = None, mode='train'):
		# seq shape: [seq_length, batch_size]
		if mode=='train':
			mask = self._generate_square_subsequent_mask(len(seq)).to(self.args.device)
		else:
			mask = None
		#print (mask)
		#print (tgt_padding_masking)
		# seq = self.encoder(seq) * math.sqrt(self.nhidden)
		# seq = self.pos_encoder(seq)
		#print ("before decoder. seq:", seq.shape, "fold embed shape:", fold_embed.shape)

 		# ad hoc way to do mem key padding mask!!!!!!!!!!!
		if fold_embed.size(0)==125:
			memory_key_padding_mask=None
		else:
			memory_key_padding_mask=tgt_padding_masking
		output = self.transformer_decoder(seq, fold_embed, tgt_mask = mask, tgt_key_padding_mask = tgt_padding_masking, \
			memory_key_padding_mask=memory_key_padding_mask)
		return output



