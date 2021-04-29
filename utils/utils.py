import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.metrics import classification_report, f1_score

class Data(Dataset):
    """
        Dataset for parsing sentences and their label sequences
    """
    def __init__(self,hparams):
        # self.seq_length = hparams['seq_length']
        self.entity_list = [
    'B-award',
    'I-award',
    'B-conference',
    'I-conference',
    'B-department',
    'I-department',
    'B-location',
    'I-location',
    'B-major',
    'I-major',
    'B-name',
    'I-name',
    'B-organization',
    'I-organization',
    'B-position',
    'I-position',
    'B-scholarship',
    'I-scholarship',
    'O'
        ]
        
        tag2idx = dict()
        tag2idx['[START]'] = len(tag2idx)
        tag2idx['[END]'] = len(tag2idx)
        tag2idx['[PAD]'] = len(tag2idx)

        for item in self.entity_list:
            tag2idx[item] = len(tag2idx)

        self.tag2idx = tag2idx
        self.path = hparams['path']

        self.bert = False
        if 'bert' in hparams:
            self.bert = True
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['bert'])
        
        self.max_length = hparams['seq_length']

        self._parse_file()

    def _parse_file(self):
        """ parse the labeled training file to collect sentences and corresponding labels, and assign them 
        as the attributes of self

        Args:
            path: string of target file
        """
        f = open(self.path,'r',encoding='utf-8')
        sentences = []
        labels = []

        sentence = []
        label = []

        vocab = {'[PAD]':0}
        # max_length = 0

        for i,line in enumerate(f):
            if line == '\n':
                # append the period at the end of the sentence for better learning the partition
                # sentence.append('[SEP]')
                
                # append the extra labels
                # label.append('O')

                sentences.append(sentence)
                labels.append(label)

                # if len(sentence) > max_length:
                #     max_length = len(sentence)
                
                sentence = []
                label = []
                continue
            
            pair = line.strip().split('\t')
            sentence.append(pair[0])
            
            if pair[0] not in vocab:
                vocab[pair[0]] = len(vocab)

            if len(pair) == 1:
                label.append('O')
            elif len(pair) == 2:
                label.append(pair[1])
            else:
                print("Error when spliting a line {}, which is {}".format(i, line))
                raise ValueError
        
        if sentence:
            # append the period at the end of the sentence for better learning the partition
            # sentence.append('。')
            
            # append the extra labels
            # label.append('O')
            # label.insert(0,'[START]')
            # label.append('[END]')

            sentences.append(sentence)
            labels.append(label)

            # if len(sentence) > max_length:
            #     max_length = len(sentence)
            
            sentence = []
            label = []

        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        # [SEP] and [CLS]
        # self.max_length = max_length + 2

        if self.bert:
            for label in self.labels:
                label.insert(0,'O')
                label.append('O')
            
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self,idx):
        back_dict = {}
        sentence = self.sentences[idx]
        
        if self.bert:
            sentence_ = ''.join(sentence)
            encoded_dict = self.tokenizer.encode_plus(sentence_, pad_to_max_length=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            tokens = encoded_dict['input_ids'].squeeze()
            back_dict['attn_mask'] = encoded_dict['attention_mask'].squeeze()

        else:
            sentence_ = ''.join(sentence)
            tokens = [self.vocab[word] for word in sentence]
            tokens = tokens[:self.max_length] + [0] * (self.max_length - len(tokens))
            tokens = np.asarray(tokens)

        label = [self.tag2idx[i] for i in self.labels[idx]]
        label = label[:self.max_length] + [self.tag2idx['[PAD]']] * (self.max_length - len(label))
        
        back_dict['sentence'] = sentence_
        back_dict['token'] = tokens
        back_dict['label'] = np.asarray(label)

        return back_dict

def my_collate(data):
    """ 
        costomized collate_fn, converting data to list rather than tensor
    """
    excluded = ['sentence']
    result = defaultdict(list)
    for d in data:
        for k,v in d.items():
            result[k].append(v)
    for k,v in result.items():
        if k not in excluded:
            result[k] = torch.tensor(v)
        else:
            continue
    return dict(result)

def prepare(hparams, split=0.9):
    """ prepare dataset and dataloader for training

    Args:
        hparams: dict of hyper parameters
        split: the portion of training set
    
    Returns:
        tag2idx: the map from the name of the tag to the index of it
        loader_train: dataloader for training, without multi-process by default
    """
    dataset = Data(hparams)

    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size

    dataset_train, dataset_val = random_split(dataset,[train_size, val_size])
    
    loader_train = DataLoader(dataset_train, batch_size=hparams['batch_size'], num_workers=8, drop_last=False, pin_memory=True, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=hparams['batch_size'], num_workers=8, drop_last=False, pin_memory=True)
    if dataset.bert:
        attr_dict = {
            'tag2idx': dataset.tag2idx,
            'tokenizer': dataset.tokenizer
        }
    else:
        attr_dict = {
            'tag2idx': dataset.tag2idx,
            'tokenizer': dataset.vocab
        }

    return attr_dict, [loader_train,loader_val]

def evaluate(model, loader,  prt=True):
    """ evaluate the model by accuracy, recall and f1-score

    Args:
        model
        loader: DataLoader
    
    Returns:
        report: accuracy, recall, f1-score printed
    """
    with torch.no_grad():
        preds = []
        labels = []
        for i,x in enumerate(loader):      
            preds.extend(model(x).flatten().tolist())
            labels.extend(x['label'].flatten().tolist())
    if prt:
        print(classification_report(labels, preds))
    
    return {
        'macro_f1': round(f1_score(y_true=labels, y_pred=preds, average='macro'),4),
        'weighted_f1': round(f1_score(y_true=labels, y_pred=preds, average='weighted'),4),
        # 'micro_f1': round(f1_score(y_true=labels, y_pred=preds, average='micro'),4),
    }


def train(hparams, model, loaders, lr=1e-3,schedule=False):
    """ train the model

    Args:
        model
        loader: DataLoader
    """

    optimizer = optim.AdamW(model.parameters(),lr=lr)

    if schedule:
        total_steps = len(loaders[0]) * hparams['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    for epoch in range(hparams['epochs']):
        tqdm_ = tqdm(enumerate(loaders[0]))
        total_loss = 0

        for step,x in tqdm_:
            loss = model.fit(x)
            loss.backward()

            # prevent gradient explosion
            # torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            
            if schedule:
                scheduler.step()

            total_loss += loss.item()
            tqdm_.set_description("epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, total_loss/(step+1)))
        
        # print the performance on validation set every epoch
        print(evaluate(model, loaders[1], prt=False))
    
    return model

def predict(sentence, model, tokenizer, max_length=None):
    """
        convert the input sentence to its word ids, then feed it into the model
    
    Args: 
        sentence: list of regular string
        model: NER model
        tokenizer: vocab or bert-tokenizer
        max_length: all sentence will be padded/truncated to max_length
    
    Returns:
        result: tagging sequence
    """
    idx2tag = {v:k for k,v in model.tag2idx.items()}
    if not max_length:
        max_length = model.seq_length

    if hasattr(model, 'bert'):
        sentence = [tokenizer.encode_plus(sent, pad_to_max_length=True, truncation=True, max_length=max_length, return_tensors='pt') for sent in sentence]

        token = torch.cat([sent['input_ids'] for sent in sentence], dim=0)
        attn_masks = torch.cat([sent['attention_mask'] for sent in sentence], dim=0)

        tag_seq = model({'token':token, 'attn_mask':attn_masks}).tolist()
        tag_seq = [[idx2tag[j] for j in i] for i in tag_seq]

        original = [tokenizer.convert_ids_to_tokens(toke) for toke in token]

    else:
        original = [[word for word in sent] for sent in sentence]

        sentence = [[tokenizer[i] for i in sent] for sent in sentence]
        sentence = [sent + [0] * (max_length - len(sent)) for sent in sentence]
        sentence = torch.tensor(sentence, device=model.device, dtype=torch.long)
        tag_seq = model({'token': sentence})
        tag_seq = [[idx2tag[j] for j in i] for i in tag_seq.tolist()]
    
    for sent,tags in zip(original, tag_seq):
        print('************************')
        for word,tag in zip(sent,tags):
            print(word, tag)