* Environments:
'environment.yml'



* How to generate seqs:

go to 'src/'
`python inference.py --trained_model $path_to_the_trained_model --output $path_to_the_output_file --pdb $pdb_fold_features`

For example, $pdb_fold_features = ../data/fold_features/1ab0A00-1-131.npy


