import numpy as np
import pickle
import torch
from collections import OrderedDict
from tape import ProteinBertModel, TAPETokenizer
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import multivariate_normal
tokenizer = TAPETokenizer(vocab='iupac') 
model = ProteinBertModel.from_pretrained('bert-base')
amino_acid = ['A', 'R', 'N', 'D',  'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])

def to_intger_padding(seq, maxlen):
    vec = np.zeros((maxlen), dtype=np.int32)
    for i in range(len(seq)):
        #print (np.where(amino_acid==seq[i]), seq[i], seq)
        vec[i] = IUPAC_VOCAB[seq[i]]
    
    return vec


# seq='ACDDQW'
# print(tokenizer.encode(seq))
# print(to_intger_padding(seq, 100))

def prepocess():
	with open("domain_dict.pkl", "rb") as f:
		domain_dict = pickle.load(f)


	seq={}
	fold_dict={}
	for i in domain_dict:
		#print (i, len(domain_dict[i]))
		#print (domain_dict[i])
		
		if domain_dict[i]['seq'] not in seq:
			seq[domain_dict[i]['seq']] =i

		fold_dict[domain_dict[i]['fold']]=1
	print (len(domain_dict), len(seq), len(fold_dict))


	domain_data={}
	ind=0
	for i in seq:
		label = seq[i]

		domain_data[label] = domain_dict[label]
		#domain_data[label]['ind']=ind
		ind+=1


	print (len(domain_data))
	with open("domain_dict", "wb") as f:
		pickle.dump(domain_data,f)

	for i in domain_data:
		for aa in domain_data[i]['seq']:
			assert aa in amino_acid

	exit(0)

	fold_dict={}

	for i in domain_data:
		if domain_data[i]['fold'] not in fold_dict:
			fold_dict[domain_data[i]['fold']] = [i]
		else:
			fold_dict[domain_data[i]['fold']].append(i)

	n5=0
	n50=0

	fold_index=[]
	for i in fold_dict:
		if len(fold_dict[i])<=5:
			n5+=1
		elif len(fold_dict[i])<=50:
			n50+=1
		fold_index.append(i)

	print(n5, n50, len(fold_dict)-n5-n50, len(fold_dict))	

	np.savetxt("fold_index.txt", fold_index, fmt='%s')
	# split training and validation sets:

	for i in fold_dict:
		ind_list=[]
		for j in fold_dict[i]:
			ind_list.append(j)
		ind_list=np.array(ind_list)
		np.random.shuffle(ind_list)
		for i in range(len(ind_list)):
			if i<0.9*float(len(ind_list)):
				domain_data[ind_list[i]]['train']=True
			else:
				domain_data[ind_list[i]]['train']=False


	x=[]
	y=[]
	train_idx=[]
	test_idx=[]
	tape_x=[]

	idx=0
	for i in domain_data:
		tokenize = to_intger_padding(domain_data[i]['seq'], 500)
		temp=tokenizer.encode(domain_data[i]['seq'])
		temp=model(torch.tensor([temp]))
		temp=torch.mean(temp[0],1).squeeze()
		print(idx)

		tape_x.append(temp)
		x.append(tokenize)
		y.append(fold_index.index(domain_data[i]['fold']))
		if domain_data[i]['train']:
			train_idx.append(idx)
		else:
			test_idx.append(idx)
		idx+=1

	print (len(train_idx), len(test_idx), train_idx[0:10])


	with open("v2_fold_classification_task_data", "wb") as f:
		pickle.dump({'x':x,'y':y,"train_idx":train_idx,"test_idx":test_idx}, f)

	with open("v2_tape_fold_classification_task_data", "wb") as f:
		pickle.dump({'x':tape_x,'y':y,"train_idx":train_idx,"test_idx":test_idx}, f)

def cal_mean(coor1):
	coor=[]
	for i in coor1:
		coor.append(i['CA'])
		#print (i)
	#print (len(coor), len(coor[0]))
	return np.mean(coor, axis=0)

def rot_matrix(theta, ux, uy, uz):
	
	sint = np.sin(theta)
	cost = np.cos(theta)

	rot =[ [ cost + ux**2*(1-cost), ux*uy*(1-cost) - uz*sint, ux*uz*(1-cost)+uy*sint ] ,
	       [ uy*ux*(1-cost) + uz*sint,  cost + uy**2*(1-cost), uy*uz*(1-cost) - ux*sint],
	       [ uz*ux*(1-cost) - uy*sint,  uz*uy*(1-cost) + ux*sint, cost + uz**2*(1-cost)]
	        ]
	return np.array(rot)

def feat_gen(domain_dict):
	#for i in domain_dict:
	#	print (domain_dict[i])

	print ("start translating proteins to make the center of mass to be at the orign of the coordinate system.")
	print ("start rotating proteins to make the first CA atom on the z-axis......")
	print ("total number of proteins: ", len(domain_dict))

	maxd =[]
	for i in domain_dict:
		camean = cal_mean(domain_dict[i]['3d'])

		#print (camean, len(domain_dict[i]['3d']))
		for j in range(len(domain_dict[i]['3d'])):
			domain_dict[i]['3d'][j]['CA'] -= camean


		first_residue_ca_coor = domain_dict[i]['3d'][0]['CA']
		
		theta = np.arccos(-first_residue_ca_coor[2] / np.sqrt(np.sum(first_residue_ca_coor**2)))
		x1 = -first_residue_ca_coor[1] / np.sqrt(first_residue_ca_coor[0]**2 + first_residue_ca_coor[1]**2)
		y1 = first_residue_ca_coor[0] / np.sqrt(first_residue_ca_coor[0]**2 + first_residue_ca_coor[1]**2)

		#print (first_residue_ca_coor, theta/3.14159*180, x1, y1)
		rot = rot_matrix(theta, x1, y1, 0.)

		c1 = np.matmul(rot, first_residue_ca_coor.T)
		assert c1[0]**2 + c1[1]**2 < 1E-20

		
		for j in range(len(domain_dict[i]['3d'])):
			domain_dict[i]['3d'][j]['CA'] = np.matmul(rot, domain_dict[i]['3d'][j]['CA'].T).T

	return domain_dict



def featurization(domain_dict, keys):

	print ('start generating features......')
	box_size = 40.
	voxel_size= 2.
	std_gassuian = 2.0
	var_gassuian = std_gassuian**2 
	threshold = (std_gassuian*3)**2
	numbox = int(box_size/voxel_size)

	ss_map = {'H': 0, 'G': 0, 'I': 0, 'B': 1, 'E':1, ' ':2, 'T':3, 'S':3}
	ratio_all=[]

	for i in keys:
		new_coor=[]
		features = np.zeros((numbox, numbox, numbox, 4), dtype=float)

		for j in range(len(domain_dict[i]['3d'])):
			new_coor.append(domain_dict[i]['3d'][j]['CA'])

		minc = np.min(new_coor, axis=0)
		maxc = np.max(new_coor, axis=0)

		
		
		ratio = box_size/(maxc-minc)

		print ('ratio: ', ratio)
		ratio_all.extend(ratio)
		centerx = np.linspace(minc[0]*ratio[0]+voxel_size/2, minc[0]*ratio[0]+box_size-voxel_size/2, numbox)
		centery = np.linspace(minc[1]*ratio[1]+voxel_size/2, minc[1]*ratio[1]+box_size-voxel_size/2, numbox)
		centerz = np.linspace(minc[2]*ratio[2]+voxel_size/2, minc[2]*ratio[2]+box_size-voxel_size/2, numbox)

		#print (minc*ratio[0], maxc*ratio[0])
		
		for j in range(len(domain_dict[i]['3d'])):
			atom_pos = domain_dict[i]['3d'][j]['CA']*ratio
			#density = norm(mean=0, cov=std_gassuian)
			type_ss = ss_map[domain_dict[i]['3d'][j]['ss']]
			for i1 in range(numbox):
				for i2 in range(numbox):
					for i3 in range(numbox):

						d2square = (atom_pos[0]-centerx[i1])**2+(atom_pos[1]-centerx[i2])**2+(atom_pos[2]-centerx[i3])**2 

						# if d2square > threshold:
						# 	continue

						density = np.exp(-d2square/var_gassuian)

						features[i1][i2][i3][type_ss] +=  density

		
		np.save("fold_features/"+i.replace('/','-'), features)					
		


import sys

if __name__=='__main__':
	#prepocess()
	with open("domain_dict.pkl", "rb") as f:
		domain_dict = pickle.load(f)
	domain_dict = feat_gen(domain_dict)
	keys=[]
	for i in domain_dict:
		keys.append(i)
	if sys.argv[1]=='1':
		sub_keys=keys[0:10000]
	elif sys.argv[1] == '2':
		sub_keys=keys[10000:20000]
	elif sys.argv[1] == '3':
		sub_keys=keys[20000:30000]
	elif sys.argv[1] == '4':
		sub_keys=keys[30000:40000]
	elif sys.argv[1] == '5':
		sub_keys=keys[40000:50000]
	elif sys.argv[1] == '6':
		sub_keys=keys[50000:60000]
	else:
		sub_keys=keys[60000:]
	featurization(domain_dict, sub_keys)



