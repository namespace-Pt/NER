# BiLSTM and CRF with Fine-Tuned BERT for Named Entity Recognition

## Instruction
- To get familiar with **Conditional Random Field**, please read [THIS](docs/CRF.pdf)
- The simplified theory of the whole model is discussed [HERE](docs/Explained.pdf).

### Dataset
all data should be included in one file with the following form:
```
2	O
0	O
2	O
0	O
年	O
6	O
月	O
4	O
日	O
，	O
国	B-organization
际	I-organization
计	I-organization
算	I-organization
机	I-organization
学	I-organization
会	I-organization
（	I-organization
A	I-organization
C	I-organization
M	I-organization
）	I-organization
```
where one word together with its label (separated with `\t`) occupies one line, and different sentences are separated by `\n`. My data is [HERE](data/data.txt).

**Importantly, the training data and testing data( validating data included ) should be all packed into one .txt file, the script will automatically split the training and testing set.**

### train and test
- **First of all**, customize your data path in `hparams['path']` in `*.ipynb`
- To inspect **the regular model without Bert**, run `base.ipynb`
- To inspect **the Bert augmented model**, run `bert.ipynb`

## Features
- [x] **batched** version of official NER provided by `PyTorch` [HERE](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
- [x] Atandard Training Workflow in `PyTorch`
  1. dataset
  2. dataloader
  3. model
  4. train
  5. evaluate
- [x] **Customized Dataset**
- [x] **Bert** Augmented

## Performance
|model|Accuracy|f1-macro|f1-weighted|
|:-:|:-:|:-:|:-:|
|base|0.95|0.7378|0.9517|
|bert|**suffers**|