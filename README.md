# Fold2Seq: A Joint Sequence(1D)-Fold(3D) Embedding-based Generative Model for Protein Design

![Fold2Seq Architecture](/fold2seq1.png)

## Environment file: 
* `environment.yml`

## Data and Feature Generation:
* Go to `data/` and check the README there. 

## How to train the model:
* How to generate seqs:

## How to generate 

go to 'src/'
`python inference.py --trained_model $path_to_the_trained_model --output $path_to_the_output_file --pdb $pdb_fold_features`

For example, $pdb_fold_features = ../data/fold_features/1ab0A00-1-131.npy

## Fold2Seq generated structures against natural structures:
![Fold2Seq structures](/fold2seq3.png)
