# BiLSTM and CRF with Fine-Tuned BERT for Named Entity Recognition

## Instruction
- To get familiar with **Conditional Random Field**, please read [THIS](CRF.pdf)
- The simplified theory of the whole model is discussed [HERE](Explained.pdf).

### train and test
- **First of all**, customize your data path in `hparams['path']` in `*.ipynb`
- To inspect the regular model without Bert, run `common.ipynb`
- To inspect the Bert augmented model, run `bert.ipynb`

## Features
- [x] **batched** version of official NER provided by `PyTorch` in [HERE](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
- [x] standard training workflow in `PyTorch`
  1. dataset
  2. dataloader
  3. model
  4. train
  5. evaluate
- [x] Bert augmented

## Performance