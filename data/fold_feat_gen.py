##########################################################################
# Load pdbs  with the chain and residue constraints.
##########################################################################

import numpy as np
import os
import sys
import pickle
import determine_ss
import argparse
import ss_dense_gen

threetoone={
'ALA':'A',
'ARG':'R',
'ASN':'N',
'ASP':'D',
'CYS':'C',
'GLU':'E',
'GLN':'Q',
'GLY':'G',
'HIS':'H',
'ILE':'I',
'LEU':'L',
'LYS':'K',
'MET':'M',
'PHE':'F',
'PRO':'P',
'SER':'S',
'THR':'T',
'TRP':'W',
'TYR':'Y',
'VAL':'V',
'MSE':'M'
}

def selection(pdb_path, chain, start, end, ss):
    b=0
    jd=0
    jd1=0
    outs=[]
    start = start.replace(')','')
    start = start.replace('(','')
    seqs=''
    ca_coor=[]
    #print (start, start.strip('('))
    #exit(0)
    end = end.replace(')','')
    end = end.replace('(','')
    with open(pdb_path, "r") as f:
            for lines in f:
                if len(lines)<5 or lines[0:4]!='ATOM':
                    if b==2:
                        break
                    continue
                if lines[21]!=chain:
                    if b==2:
                        break
                    continue
                resi = lines[22:27].strip(' ')
                if b==2 and resi!=end:
                    break
                if resi==start:
                    b=1
                    jd=1
                if resi==end:
                    b=2
                    jd1=1
                elif b==2:
                    break
                if b==1 or b==2:
                        
                    if lines[13:16]=='CA ':
                        resi={'name': lines[17:20]}
                        resi['CA'] = [float(lines[30:38]), float(lines[38:46]), float(lines[46:54])]
                        ca_coor.append(resi)
                        seqs+=threetoone[lines[17:20]]
    if jd*jd1!=1:
        raise ValueError("encounter inconsistent pdb structure:"+pdb_path+chain+" "+start+','+end)


    start,end = dp(seqs, ss['seq'])

    j=start
    for i in range(len(ca_coor)):
        while ss['seq'][j]!=threetoone[ca_coor[i]['name']]:
            j+=1
        ca_coor[i]['ss'] = ss['ss'][j]

    return ca_coor, ss['seq'][start:end]


def dp(cst, cseq):

    k=0
    best_start=0
    best_end=0
    best_score=10000
    while k<len(cseq)-len(cst)+1:
        
        if cseq[k] == cst[0]:
            i=0
            j=k

            while i<len(cst) and j<len(cseq):
                if cst[i]==cseq[j]:
                    i+=1
                j+=1

            if i==len(cst):
                if j-k< best_score:
                    best_score=j-k
                    best_start=k
                    best_end=j
                    if best_score == len(cst):
                         break
        k+=1

    if best_score==10000:
        print(cst)
        print(cseq)
        raise ValueError("do not find alignment.")


    return best_start, best_end


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Arguments for pretrain_seq.py')
    parser.add_argument('--domain_list', default='./domain_list.txt', type=str)
    parser.add_argument('--out', default='./domain_dict.pkl', type=str)
    parser.add_argument('--ss', default='./ss.txt', type=str)
    args = parser.parse_args()


    domain_seq={}
   
    seq_ss = determine_ss.read_ss(args.ss)
    num=0

    with open(args.domain_list, "r") as f:
      for lines in f:
        line = lines.strip('\n').split()
        label = line[1].upper()+':'+line[2]
        i = line[1]+line[2]+'00-'+line[3]+'-'+line[4]

        if label in seq_ss:
            
            x1,x2 = selection(line[0] ,  line[2], (line[3]), (line[4]), seq_ss[label])  

            domain_seq[i]={}
            domain_seq[i]['seq'] =x2
            domain_seq[i]['3d'] = x1


        print ("processed seqs: %d" %(num))
        num+=1
    keys=[]
    for i in domain_seq:
        keys.append(i)
    ss_dense_gen.featurization(domain_seq, keys)
    with open(args.out, "wb") as f:
        pickle.dump(domain_seq, f)


