import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse
import numpy as np
import math
import os
import pickle
import numpy as np
import random
from Utils import amino_acid
from Utils import model_statistics
import seq_decoder
import fold_encoder
class ProteinSeqTransformer(nn.Module):

    def __init__(self, args):
        #  nhidden:  the size of hidden state
    
        super(ProteinSeqTransformer, self).__init__()

        self.model_type = 'ProteinSeqTransformer'
        
        encoder_layers = TransformerEncoderLayer(args.nhidden, args.nhead, args.seq_encoder_feedforward, args.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers)


    def forward(self, src, padding_masking=None):


        output = self.transformer_encoder(src, src_key_padding_mask = padding_masking)

        return output

class cosine_similarity(nn.Module):
    def __init__(self, args):
        super (cosine_similarity, self).__init__()
        self.row_wise_avgpool = nn.AvgPool1d(kernel_size = 3, stride=1)

    def forward(self, x, y):
        # x shape [len_x, bsz, hidden] seq
        # y shape [len_y, bsz, hidden] fold
        x = x.transpose(0,1)
        y = y.transpose(0,1)
        #print (x,y)
        cos_mat = torch.softmax(torch.bmm(y, x.transpose(1,2)), dim=2) # cos_mat shape [bsz, len_y, len_x]
        #print (x.shape, y.shape, cos_mat.shape, cos_mat)
        cos_mat = self.row_wise_avgpool(cos_mat)
        #print (cos_mat.shape)
        cos_vec, t = torch.max(cos_mat, dim=2)
        cos_vec = torch.mean(cos_vec, dim=1)
        print ("cos_vec", cos_vec.shape)
        return cos_vec
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=502):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def zero_padding(hidden, padding_masking):
    c = 1-padding_masking.int()
    #print (c[0], padding_masking[0])
    hidden*=c.transpose(0,1).unsqueeze(-1)



class fold_classification_generator(nn.Module):
    def __init__(self, args):
        super(fold_classification_generator, self).__init__()

        self.args = args

        #self.lineartape_to_hidden = nn.Linear(768, args.nhidden)
        
        #self.linear1 = nn.Linear(args.nhidden, args.nfolds)
        #-------------fold encoder------------------------------------------
        ####################################################################
        if args.lba0 !=0 or args.lba2 !=0 or args.lba4!=0:
        	self.fold_encoder = fold_encoder.fold_encoder(args)

        #-------------seq encoder------------------------------------------
        ####################################################################
        if args.lba1 !=0 or args.lba3 !=0 or args.lba4!=0:
            self.seq_encoder = ProteinSeqTransformer(args)
            self.seq_embedding = nn.Embedding(args.ntokens, args.nhidden)
            self.positional_embedding = PositionalEncoding(args.nhidden,args.dropout)

        #-------------seq decoder------------------------------------------
        ####################################################################
        if args.lba0 !=0 or args.lba1 !=0:
            self.seq_decoder = seq_decoder.transformer_decoder(args)
            self.seq_embedding = nn.Embedding(args.ntokens, args.nhidden)
            self.positional_embedding = PositionalEncoding(args.nhidden, args.dropout)
            self.decoder_out_linear = nn.Linear(args.nhidden, args.ntokens-2)
        #-------------similar measure------------------------------------------
        ####################################################################        
        if args.lba4!=0:
        	self.cosine_similarity = cosine_similarity(args)

        #------------- fold/seq classification------------------------------------------
        ####################################################################
        if args.lba2!=0 or args.lba3!=0:
            self.fold_classification_linear = nn.Linear(args.nhidden, args.nfolds)
        
        
        self.dropout = nn.Dropout(args.dropout) 


    def forward(self, seq, fold, padding_masking=None, mode='train'):
        # seq: shape: [batch_size, seq_len]
        # fold shape [bsz, 20, 20, 20, 4]
        token_preds_from_fold=None
        token_preds_from_seq=None
        foldclass_preds=None
        seqclass_preds=None
        sim_score0=None
        sim_score1=None
        seqclass_preds1=None


        seq=seq.transpose(0,1)

        seq = self.seq_embedding(seq)
        seq = self.positional_embedding(seq)

        #-------------fold encoder------------------------------------------
        ####################################################################
        if self.args.lba0+self.args.lba2+self.args.lba4!=0:
            hidden_state_fold = self.fold_encoder(fold)  # hidden: [seq_len, batch_size, hidden states]
            mean_hidden_state_fold = torch.mean(hidden_state_fold, dim=0)

        #-------------seq encoder------------------------------------------
        ####################################################################
        if self.args.lba1+self.args.lba3+self.args.lba4!=0:
            hidden_state_seq = self.seq_encoder(seq, padding_masking=padding_masking)
            #print (hidden_state_seq, 'sdd')  
            zero_padding(hidden_state_seq, padding_masking)
            #print (hidden_state_seq)
            mean_hidden_state_seq = torch.mean(hidden_state_seq, dim=0)

        #-------------fold2seq ------------------------------------------
        ####################################################################
        if self.args.lba0!=0:
            decoder_out_from_fold = self.seq_decoder(seq, hidden_state_fold, padding_masking, mode)
            token_preds_from_fold = self.decoder_out_linear(decoder_out_from_fold).transpose(0,1)

        #-------------seq2seq decoder------------------------------------------
        ####################################################################
        if self.args.lba1!=0:
            decoder_out_from_seq = self.seq_decoder(seq, hidden_state_seq, padding_masking, mode)
            token_preds_from_seq = self.decoder_out_linear(decoder_out_from_seq).transpose(0,1)  
        
        #-------------fold2 fold classification------------------------------------------
        ####################################################################
        if self.args.lba2!=0:        
            foldclass_preds = self.fold_classification_linear(mean_hidden_state_fold)

        #-------------seq2 fold classification------------------------------------------
        ####################################################################
        if self.args.lba3!=0:
            seqclass_preds = self.fold_classification_linear(mean_hidden_state_seq)           

        #-------------similarity measure------------------------------------------
        ####################################################################
        if self.args.lba4!=0:
            #sim_score = torch.sum((mean_hidden_state_seq - mean_hidden_state_fold)**2, dim=1)/(self.args.nhidden**2)
            sim_score0 = -self.cosine_similarity(hidden_state_seq, hidden_state_fold)
            pred_seq = torch.max(token_preds_from_fold.transpose(0,1)  , dim=2).indices  # shape [ seq_len, bsz]
            start = torch.ones((1, pred_seq.size(1)) , dtype=torch.long, device=self.args.device)+20
            pred_seq = torch.cat( (start, pred_seq[:-1]), dim=0 )
            pred_seq = self.seq_embedding(pred_seq.long())
            pred_seq = self.positional_embedding(pred_seq)
            hidden_state_seq1 = self.seq_encoder(pred_seq, padding_masking=padding_masking)
            #print (hidden_state_seq, 'sdd')  
            zero_padding(hidden_state_seq1, padding_masking)
            #print (hidden_state_seq)
            mean_hidden_state_seq1 = torch.mean(hidden_state_seq1, dim=0)
            seqclass_preds1 = self.fold_classification_linear(mean_hidden_state_seq1)
            sim_score1 = torch.mean((hidden_state_seq1 - hidden_state_seq)**2, dim=[0,2])

            #print("simscore shape", sim_score0.shape, sim_score1.shape, seqclass_preds1.shape)

        return token_preds_from_fold, token_preds_from_seq, foldclass_preds,seqclass_preds, sim_score0,sim_score1, seqclass_preds1

    

