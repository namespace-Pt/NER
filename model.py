import torch
import torch.nn as nn

START_TAG = 0
STOP_TAG = 1
class BiLSTM_CRF(nn.Module):
    '''
        BiLSTM with CRF for Named Entity Recognition
    '''
    def __init__(self, hparams, tag2idx):
        super().__init__()

        self.batch_size = hparams['batch_size']
        self.embedding_dim = hparams['embedding_dim']
        self.hidden_dim = hparams['hidden_dim']
        self.vocab_size = hparams['vocab_size']
        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag2idx[START_TAG], :] = -10000
        self.transitions.data[:, tag2idx[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _lstm_encoder(self, sentence):
        """ encode sentence with BiLSTM
        
        Args:
            sentence: word index sequence of [batch_size, seq_length]

        Returns:
            lstm_feats: sentence embedding of [batch_size, seq_length, tagset_size]
        """
        self.hidden = self.init_hidden()
        embedding = self.embedding(sentence).view(len(sentence), self.batch_size, self.embedding_dim)
        lstm_out, self.hidden = self.lstm(embedding, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.batch_size, self.hidden_dim).transpose(0,1)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
        
    def _forward_alg(self, feats):
        """ calculate the log-sum-exp of the score of all possible label sequence

        Args:
            feats: sentence embedding of [batch_size, seq_length, tagset_size]
        
        Returns:
            alpha: batch of log-sum-exp of [batch_size, 1]
        """

        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.batch_size, 1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[:, 0, self.tag2idx[START_TAG]] = 0.
        # Wrap in a variable so that we will get automatic backprop
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

    def _viterbi_decode(self, feats):
        """ find the best label sequence when inference

        Args:
            feats: sentence embedding of [batch_size, seq_length, tagset_size]
        
        Returns:
            path_score: batch of score of the input feats, of size [batch_size, 1]
            best_path: list of transmition trace
        """
        backpointers = []

        init_vvars = torch.full((self.batch_size, 1, self.tagset_size), -10000.)
        init_vvars[:, 0, self.tag2idx[START_TAG]] = 0.

        # forward_var at step i holds the viterbi variables for step i-1
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
        return path_score, best_path

    def _score_sentence(self, feats, tags):
        """ Score the provided label sequence

        Args:
            feats: sentence embedding of [batch_size, seq_length, tagset_size]
            tags: label sequence of [batch_size, seq_length]
        
        Returns:
            scores: batch of scores, size of [batch_size, 1]
        """
        # Gives the score of a provided tag sequence
        score = torch.zeros((self.batch_size,1))
        tags = torch.cat([torch.full((self.batch_size, 1), self.tag2idx[START_TAG], dtype=torch.long), tags],dim=1)
        for i in range(feats.shape[0]):
            feat = feats[:,i,:]
            # calculate score in batch
            score = score + self.transitions[tags[:,i+1], tags[:,i]].view(self.batch_size,1) + feat.gather(dim=-1, index=tags[:,i].unsqueeze(dim=-1))
        
        score = score + self.transitions[self.tag2idx[STOP_TAG], tags[:,-1]]
        return score
    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._lstm_encoder(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._lstm_encoder(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq