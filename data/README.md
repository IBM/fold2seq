# Fold2Seq Data and Feature Generation

![Fold2Seq Architecture](/data/fold2seq2.png)

## Data
The CATH IDs of protein domains in training, validation and two test sets are in `pdb_lists/`. 


## Feature Generation:
### Input File:
* In order to generate SSE density features, you need to first provide a file with all input proteins' information.  Each row describes a protein domain. The meaning of each column is:
  * Column1: The path to the PDB
  * Column2: The PDB ID
  * Column3: The chain ID
  * Column4: The starting residue ID
  * Column5: The ending residue ID
* An example of this input file is `example/domain_list.txt`.

### Secondary Structure Assignment:
* Moreover, you need to pre-assign a secondary structure element to each residue. We provide an assignment file (ss.txt) obtained from RCSB PDB which contains most of exsiting PDBs. You can first check if your protein is in this file. If not, you can append it following the format in the file.  

### Generating features:
* To generate SSE density features, you can run:

`python fold_feat_gen.py --domain_list example/domain_list.txt  --ss ss.txt --out $path_to_the_output_file`.

* It will generate a  python dictionary containing input information and fold features in `fold_features/`.

