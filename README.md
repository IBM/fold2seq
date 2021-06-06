# Fold2Seq: A Joint Sequence(1D)-Fold(3D) Embedding-based Generative Model for Protein Design

![Fold2Seq Architecture](/fold2seq1.png)

## Environment file: 
* `environment.yml`

## Data and Feature Generation:
* Go to `data/` and check the README there. 

## How to train the model:
* go to `src/` and run:

`python train.py --data_path $path_to_the_data_dictionary --lr $learning_rate --model_save $path_to_the_saved_model`

## How to generate sequences:
* go to 'src/' and run:

`python inference.py --trained_model $path_to_the_trained_model --output $path_to_the_output_file --data_path $path_to_the_data_dictionary`


## Fold2Seq generated structures against natural structures:
![Fold2Seq structures](/fold2seq3.png)
