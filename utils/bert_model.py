import torch
import torch.nn as nn
from transformers import AutoModel

START_TAG = '<START>'
STOP_TAG = '<END>'
PAD_TAG = '<PAD>'

class BiLSTM_CRF_BERT(nn.Module):
    '''
        BiLSTM with CRF for Named Entity Recognition
    '''

    def __init__(self, hparams, tag2idx):
        super().__init__()

        self.batch_size = hparams['batch_size']
        self.embedding_dim = hparams['embedding_dim']
        self.hidden_dim = hparams['hidden_dim']
        self.vocab_size = hparams['vocab_size']
        self.seq_length = hparams['seq_length']

        self.device = hparams['device']

        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)
        self.idx2tag = {v:k for k,v in tag2idx.items()}

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, bidirectional=True)

        self.bert = AutoModel.from_pretrained(
            hparams['bert'],
            # output hidden embedding of each transformer layer
            output_hidden_states=True
        )

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag2idx[START_TAG], :] = -10000
        self.transitions.data[:, tag2idx[STOP_TAG]] = -10000

        self.transitions.data[:, tag2idx[PAD_TAG]] = 0
        self.transitions.data[tag2idx[PAD_TAG], :] = 0

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2, device=self.device),
                torch.randn(2, self.batch_size, self.hidden_dim // 2, device=self.device))

    def _lstm_encoder(self, sentence, attn_mask):
        """ encode sentence with BiLSTM and BERT, use the output of the second-to-last hidden layer
            as the hidden states for each token

        Args:
            sentence: word index sequence of [batch_size, seq_length]
            attn_mask: attention mask vector

        Returns:
            lstm_feats: sentence embedding of [batch_size, seq_length, tagset_size]
        """
        self.hidden = self.init_hidden()

        output = self.bert(sentence, attn_mask)
        embedding = output['hidden_states'][-2].transpose(0,1)

        lstm_out, self.hidden = self.lstm(embedding, self.hidden)
        lstm_out = lstm_out.transpose(0,1)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        """ calculate the log-sum-exp of the score of all possible label sequence

        Args:
            feats: sentence embedding of [batch_size, seq_length, tagset_size]

        Returns:
            alpha: batch of log-sum-exp of [batch_size, 1]
        """

        init_alphas = torch.full((self.batch_size, 1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[:, 0, self.tag2idx[START_TAG]] = 0.
        forward_var = init_alphas

        # Iterate through the sentence
        for i in range(feats.shape[1]):
            feat = feats[:,i,:]

            emit_score = feat.view(self.batch_size, self.tagset_size, 1)
            next_tag_var = forward_var + self.transitions + emit_score
            forward_var = torch.logsumexp(next_tag_var,dim=-1).view(self.batch_size, 1, self.tagset_size)

        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        alpha = torch.logsumexp(terminal_var,dim=-1)
        return alpha

    def _score_sentence(self, feats, tags):
        """ Score the provided label sequence

        Args:
            feats: sentence embedding of [batch_size, seq_length, tagset_size]
            tags: label sequence of [batch_size, seq_length]

        Returns:
            scores: batch of scores, size of [batch_size, 1]
        """
        score = torch.zeros((self.batch_size,1), device=self.device)
        tags = torch.cat([torch.full((self.batch_size, 1), self.tag2idx[START_TAG], dtype=torch.long, device=self.device), tags],dim=1)
        for i in range(feats.shape[0]):
            feat = feats[:,i,:]

            score = score + self.transitions[tags[:,i+1], tags[:,i]].view(self.batch_size,1) + feat.gather(dim=-1, index=tags[:,i].unsqueeze(dim=-1))
        
        score = score + self.transitions[self.tag2idx[STOP_TAG], tags[:,-1]].unsqueeze(dim=-1)

        return score

    def neg_log_likelihood(self, x):
        sentence = x['token'].to(self.device)
        tags = x['label'].to(self.device)
        attn_mask = x['attn_mask'].to(self.device)

        if sentence.shape[0] != self.batch_size:
            self.batch_size = sentence.shape[0]

        feats = self._lstm_encoder(sentence, attn_mask)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score, dim=0)

    def _viterbi_decode(self, feats):
        """ find the best label sequence when inference

        Args:
            feats: sentence embedding of [batch_size, seq_length, tagset_size]

        Returns:
            path_score: batch of score of the input feats, of size [batch_size, 1]
            best_path: list of transmition trace
        """
        backpointers = []

        init_vvars = torch.full((self.batch_size, 1, self.tagset_size), -10000.).to(self.device)
        init_vvars[:, 0, self.tag2idx[START_TAG]] = 0.

        forward_var = init_vvars
        for i in range(feats.shape[1]):
            feat = feats[:,i,:]

            next_tag_var = forward_var + self.transitions + feat.view(self.batch_size, self.tagset_size, 1)
            best_tag_var, best_tag_id = torch.max(next_tag_var, dim=-1)
            backpointers.append(best_tag_id)
            forward_var = best_tag_var.view(self.batch_size, 1, self.tagset_size)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        path_score, best_tag_id = torch.max(terminal_var, dim=-1)

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            # gather the value in bptrs_t according to best_tag_id
            best_tag_id = bptrs_t.gather(dim=-1,index=best_tag_id)
            best_path.append(best_tag_id)
        
        # Pop off the start tag (we dont want to return that to the caller)
        best_path.pop()
        best_path.reverse()
        best_path = torch.cat(best_path, dim=1)
        # best_path = [[self.idx2tag[j] for j in i] for i in best_path.tolist()]
        
        return path_score, best_path
    
    def forward(self, sentence, attn_mask):
        sentence = sentence.to(self.device)
        attn_mask = attn_mask.to(self.device)

        if sentence.shape[0] != self.batch_size:
            self.batch_size = sentence.shape[0]

        lstm_feats = self._lstm_encoder(sentence, attn_mask)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

class BERT_NER(nn.Module):
    """
        fine-tune pretrained BERT-NER model on our dataset
    """

    def __init__(self, hparams, tag2idx):
        super().__init__()

        self.batch_size = hparams['batch_size']
        self.seq_length = hparams['seq_length']

        self.device = hparams['device']

        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)
        self.idx2tag = {v:k for k,v in tag2idx.items()}

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, bidirectional=True)

        self.bert = AutoModel.from_pretrained(
            'ckiplab/albert-base-chinese-ner',
            # output hidden embedding of each transformer layer
        )