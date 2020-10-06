def hparam_pretrain_seq():
	hparam={}
	hparam['ntoken']=20
	hparam['nhidden']=128
	hparam['nfeedfoward']=512
	hparam['nlayers'] =4
	hparam['dropout'] = 0.5

	hparam['seq_embedding'] = 'ProteinSeqTransformer'

	