import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset,DataLoader

class Data(Dataset):
    """
        Dataset for parsing sentences and their label sequences
    """
    def __init__(self,hparams):
        # self.seq_length = hparams['seq_length']
        self.refer_dict = {
            "a":"O",    #Others 无关
            "b":"B-C",  #Conference 会议、讲座、刊物的开始
            "c":"I-C",  #Conference 会议、讲座、刊物的中间
            "d":"B-A",  #Award 奖项的开始
            "e":"I-A",  #Award 奖项的中间
            "f":"B-O",  #Organization 组织的开始
            "g":"I-O",  #Organization 组织的中间
            "h":"B-M",  #Major 专业、课程的开始   
            "i":"I-M",  #Major 专业、课程的中间
            "j":"B-P",  #Position 职位、职称的开始
            "k":"I-P",  #Position 职位、职称的中间
            "l":"B-N",  #Name 人名的开始
            "m":"I-N",  #Name 人名的中间
            "n":"B-D",  #Department 学校、学院、系别的开始
            "o":"I-D",  #Department 学校、学院、系别的中间
            "p":"B-S",  #Scholarship 项目、夏令营、考试等学生活动的开始
            "q":"I-S",  #Scholarship 项目、夏令营、考试等学生活动的中间
            "r":"B-L",  #地点的开始
            "s":"I-L",  #地点的中间
            "t":"O",  #论文的开始
            "u":"O"   #论文的中间
        }
        tag2idx = dict()
        tag2idx['<START>'] = len(tag2idx)
        tag2idx['<END>'] = len(tag2idx)
        tag2idx['<PAD>'] = len(tag2idx)

        for item in self.refer_dict.values():
            if item not in tag2idx:
                tag2idx[item] = len(tag2idx)

        self.tag2idx = tag2idx

        self.path = 'D:/Data/NER/corpus/labeled_train.txt'
        # self.path = hparams['path']

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

        vocab = {'<PAD>':0, '。':1}
        max_length = 0

        for i,line in enumerate(f):
            if line == '\n':
                # append the period at the end of the sentence for better learning the partition
                sentence.append('。')
                
                # append the extra labels
                label.append('O')
                label.insert(0,'<START>')
                label.append('<END>')

                sentences.append(sentence)
                labels.append(label)

                if len(sentence) > max_length:
                    max_length = len(sentence)
                
                sentence = []
                label = []
                continue
            
            pair = line.strip().split(' ')
            sentence.append(pair[0])
            
            if pair[0] not in vocab:
                vocab[pair[0]] = len(vocab)

            if len(pair) == 1:
                label.append('O')
            elif len(pair) == 2:
                label.append(self.refer_dict[pair[1]])
            else:
                print("Error when spliting a line")
                raise ValueError

        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self,idx):
        back_dict = {}
        sentence = [self.vocab[word] for word in self.sentences[idx]]
        label = [self.tag2idx[i] for i in self.labels[idx]]
        
        sentence = sentence + [0] * (self.max_length - len(sentence))
        # <START> and <END> are extra labels
        label = label + [-1] * (self.max_length - len(label) + 2)

        back_dict['sentence'] = np.asarray(sentence)
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
            if k not in excluded:
                result[k].append(torch.LongTensor(v))
            else:
                result[k].append(v)

    return dict(result)

def prepare(hparams):
    """ prepare dataset and dataloader for training

    Args:
        hparams: dict of hyper parameters
    
    Returns:
        tag2idx: the map from the name of the tag to the index of it
        loader_train: dataloader for training, without multi-process by default
    """
    dataset = Data(hparams)
    # loader_train = DataLoader(dataset, batch_size=hparams['batch_size'], num_workers=0, drop_last=False, collate_fn=my_collate)
    loader_train = DataLoader(dataset, batch_size=hparams['batch_size'], num_workers=0, drop_last=False, pin_memory=True)
    return dataset.tag2idx, dataset.vocab, loader_train