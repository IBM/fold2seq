import pickle
import os

amino_acid = ['A', 'R', 'N', 'D',  'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def read_ss(path):
	seq_ss={}
	b=0
	with open(path, "r") as f:
		for lines in f:
			if lines[0]=='>':
				if 'sequence' in lines:
					seq_ss[lines[1:7]] = {}
					label = lines[1:7]
					seq_ss[label]['seq']=''
					seq_ss[label]['ss']=''
					b=0
				elif 'secstr' in lines:
					b=1
				else:
					raise ValueError("error!")
			else:
				if b==0:
					seq_ss[label]['seq']+=lines.strip('\n')
				else:
					seq_ss[label]['ss']+=lines.strip('\n')
	print ("total number of seqs:", len(seq_ss))

	remove = []
	for i in seq_ss:
		assert len(seq_ss[i]['seq'])==len(seq_ss[i]['ss'])
		for j in seq_ss[i]['seq']:
			if j not in amino_acid:
				remove.append(i)
				break
	for i in remove:
		del seq_ss[i]

	print ("removed # seqs in ss.txt:", len(remove))
	return seq_ss

def test_seq_identity(seq_exp,seq_ref):
	for i in range(len(seq_ref)-len(seq_exp)+1):

		if seq_exp == seq_ref[i: i+len(seq_exp)]:
			return 1

	return 0

if __name__=='__main__':

	seq_dict_new={}

	with open("seq_dict.pkl", "rb") as f:
		seq_dict = 	pickle.load(f)

	seq_ss = read_ss("ss.txt")

	n1=0
	n2=0
	n3=0
	for i in seq_dict:
		label = i[0:4].upper()+':'+i[4]

		if label in seq_ss:
			n1+=1

			if label+".pdb" not in os.listdir("pdbs/"):
				os.system("curl https://files.rcsb.org/download/"+i[0:4]+".pdb > 1" )
				os.system("grep ATOM 1 > pdbs/"+i[0:4]+".pdb")

		else:
			#print (label)
			n2+=1
	print(n1,n2, n1+n2, n3)
