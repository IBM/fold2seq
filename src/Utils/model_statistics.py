import torch
import numpy as np
def net_param_num(model):
	num=0
	for i in model.parameters():
		v=1
		for j in i.shape:
			v*=j

		num+=v
	return num

def multi_class_accuracy(target, preds_score):
	top1=0.
	top3=0.
	top5=0.
	top10=0.

	preds_score = preds_score.detach().cpu().numpy()
	target = target.cpu().numpy()

	for i in range(len(preds_score)):
		ind_sorted = np.argsort(-preds_score[i])

		if target[i] in ind_sorted[:1]:
			top1+=1
		if target[i] in ind_sorted[:3]:
			top3+=1
		if target[i] in ind_sorted[:5]:
			top5+=1
		if target[i] in ind_sorted[:10]:
			top10+=1

	#print ("top1 acc: %.4f" %(top1/len(preds_score)))
	#print ("top3 acc: %.4f" %(top3/len(preds_score)))
	#print ("top5 acc: %.4f" %(top5/len(preds_score)))
	#print ("top10 acc: %.4f" %(top10/len(preds_score)))

	return top1,top3,top5,top10
