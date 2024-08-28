# ECMM(ESM enzyme Classifier with MLP Mixer)
1. the model uses META's ESM as the embedding model and Google's MLP mixer's architecture as the enzyme classifier 
2. the purpose of the model is to look at the FASTA file and classify the enzymes 
3. we have prepared the code to train it, so let's train it!
## How to use

### Create a virtual environment

```
cd ECMM
conda env create -f environment.yaml
pip install -e .
```

### fastafiles

There are data examples in data folder.

The data has the following conditions

1. fasta_dir must not contain subfolders 
2. only one sequence must exist in the fasta file 
3. the top line of the fasta file must contain the number (integer) of the corresponding group.


### training

You can do training by writing down the conditions in train.sh and executing src/train.sh.

