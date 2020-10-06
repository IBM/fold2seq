##########################################################################
# Download a pdb file with the chain and residue constrains.
##########################################################################

import numpy as np
import os
import sys
import pickle
import determine_ss

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

def selection(pdb_path, chain, start, end, output):
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
    with open(output, "w") as fout:
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


    #print(seqs)
    start,end = dp(seqs, ss['seq'])

    j=start
    for i in range(len(ca_coor)):
        while ss['seq'][j]!=threetoone[ca_coor[i]['name']]:
            j+=1
        ca_coor[i]['ss'] = ss['ss']

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

    print(best_score - len(cst))

    return best_start, best_end

# def read_domain(path, ss):
#     seqs=''
#     ca_coor=[]
#     with open(path, "r") as f:
#         for lines in f:
#             if lines[13:16]=='CA ':
#                 resi={'name': lines[17:20]}
#                 resi['CA'] = [float(lines[30:38]), float(lines[38:46]), float(lines[46:54])]
#                 ca_coor.append(resi)
#                 seqs+=threetoone[lines[17:20]]
#     print(seqs)
#     start,end = dp(seqs, ss['seq'])

#     j=start
#     for i in range(len(ca_coor)):
#         while ss['seq'][j]!=threetoone[ca_coor[i]['name']]:
#             j+=1
#         ca_coor[i]['ss'] = ss['ss']

#     return ca_coor, ss['seq'][start:end]

# def download_sele(pdb):
#     pdb = pdb.upper()
#     link = 'https://files.rcsb.org/download/%s.pdb'%pdb

#     #sele='chain '+chain+' and resi '+start+'-'+end

#     os.system("curl %s > temp.pdb" %(link))
#     # pymol=[]
#     # pymol.append("load temp.pdb")
#     # pymol.append("select "+sele)
#     # pymol.append("save "+save_path+", sele")
#     # np.savetxt("load_pdb.pml", pymol, fmt="%s")
#     # os.system("/Applications/PyMOL.app/Contents/MacOS/PyMOL -cq load_pdb.pml")
#     #os.system("rm temp.pdb")
#     #os.system("rm load_pdb.pml")

if __name__=='__main__':
    with open("seq_dict.pkl", "rb") as f:
        seq_dict = pickle.load(f)
    seq_dict['4g50A01/-77--1']['start']='-77'
    seq_dict['4g50A01/-77--1']['end']='-1'
    del(seq_dict['4z54B00/6-291'])
    del(seq_dict['4ztbA01/1-135']) 
    del(seq_dict['4ztbA02/136-321'])
    del(seq_dict['5dejA00/10-125'])
    del(seq_dict['5fojA00/1-131']) 

    domain_seq={}
   
    seq_ss = determine_ss.read_ss("ss.txt")
    num=0
    for i in seq_dict:
        label = i[0:4].upper()+':'+i[4]
        print(i)
        if label in seq_ss:
            #seq_dict_new[i] = seq_dict[i]
         
            #x1,x2 = read_domain("domains/"+i.replace('/','-'), seq_ss[label])
            x1,x2 = selection("pdbs/"+seq_dict[i]['pdb']+'.pdb' , seq_dict[i]['chain'],\
                 seq_dict[i]['start'], seq_dict[i]['end'], "domains/"+i.replace('/','-'))  

            domain_seq[i]={}
            domain_seq[i]['seq'] =x2
            domain_seq[i]['3d'] = x1
            domain_seq[i]['fold'] = seq_dict[i]['fold']


        print ("processed seqs: %d/%d" %(num, len(seq_dict)))
        num+=1

    with open("domain_dict.pkl", "wb") as f:
        pickle.dump(domain_seq, f)


