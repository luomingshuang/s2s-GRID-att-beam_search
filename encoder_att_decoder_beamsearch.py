import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import pandas as pd


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):

        outputs, hidden = self.gru(src, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn1 = nn.Linear(self.hidden_size * 2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1) 
    
    def _init_hidden(self):
        nn.init.xavier_normal_(self.attn1.weight)
        nn.init.xavier_normal_(self.attn2.weight)

    def forward(self, hidden, encoder_outputs): 
        seq_len, batch_size, _ = encoder_outputs.size()
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        inputs = torch.cat((encoder_outputs, h), 2).view(-1, self.hidden_size*2)
        o = self.attn2(F.tanh(self.attn1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)
        return context

    # def score(self, hidden, encoder_outputs):
    #     # [B*T*2H]->[B*T*H]
    #     assert(hidden.size(2) + encoder_outputs.size(2) == self.hidden_size*2)
    #     energy = self.attn(torch.cat([hidden, encoder_outputs], 2)).softmax(-1)
    #     energy = energy.transpose(1, 2)  # [B*H*T]
    #     v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
    #     #print('')
    #     energy = torch.bmm(v, energy)  # [B*1*T]
    #     return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        # print('last_hidden size is:', last_hidden.size())>>>[3, 16, (1024)hidden_size]
        # print('last_hidden[-1] size is:', last_hidden[-1].size())>>>[1, 16, 1024]
        context = self.attention(last_hidden[-1], encoder_outputs)
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        # context = context.transpose(0, 1)  # (1,B,N)
        #print('context size is:', context.size())
        # Combine embedded input word and attended context, run through RNN
        #print('embedded size is:', embedded.size())
        context = context.transpose(0, 1)
        #print(embedded.size(), context.size())
        rnn_input = torch.cat([embedded, context], 2)
        #print('rnn_input size is:', rnn_input.size())
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, beam_size=5):
        batch_size = src.size(1)
        max_len = 2
        if(not trg is None):
            max_len = trg.size(0)
        else:
            max_len = src.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]

        output = torch.zeros(src.size(1)).long().cuda()  # sos

        decoder_outputs, decoder_hidden = self.decoder(output, hidden, encoder_output)
        topk = decoder_outputs.data.topk(beam_size)
        samples = [[] for i in range(beam_size)]
        dead_k = 0
        final_samples = []
        for index in range(beam_size):
            topk_prob = topk[0][0][index]
            topk_index = int(topk[1][0][index])
            samples[index] = [[topk_index], topk_prob, 0, 0, decoder_hidden, encoder_output]
       # print(samples)
        for _ in range(max_len):
            tmp = []
            for index in range(len(samples)):
                tmp.extend(self.beamSearchInfer(samples[index], index, beam_size))
            samples = []

            # 筛选出topk
            df = pd.DataFrame(tmp)
            df.columns = ['sequence', 'pre_socres', 'fin_scores', "ave_scores",  "decoder_hidden", "encoder_outputs"]
            sequence_len = df.sequence.apply(lambda x:len(x))
            df['ave_scores'] = df['fin_scores'] / sequence_len
            df = df.sort_values('ave_scores', ascending=False).reset_index().drop(['index'], axis=1)
            df = df[:(beam_size-dead_k)]
            for index in range(len(df)):
                group = df.ix[index]
                if group.tolist()[0][-1] == 1:
                    final_samples.append(group.tolist())
                    df = df.drop([index], axis=0)
                    dead_k += 1
                    print("drop {}, {}".format(group.tolist()[0], dead_k))
            samples = df.values.tolist()
            if len(samples) == 0:
                break

        if len(final_samples) < beam_size:
            final_samples.extend(samples[:(beam_size-dead_k)])

        #print(final_samples)
        return final_samples

        
        # for t in range(1, max_len):
        #     output, hidden = self.decoder(
        #             output, hidden, encoder_output)
        #     #print('decoder hidden size is:', hidden.size())
        #     outputs[t] = output
        #     if(not trg is None):
        #         is_teacher = random.random() < teacher_forcing_ratio
        #     else:
        #         is_teacher = False
        #     top1 = output.data.max(1)[1]
        #     output = Variable(trg.data[t] if is_teacher else top1).cuda()
        # return outputs

    def beamSearchInfer(self, sample, k, beam_size):
        samples = []
        decoder_input = Variable(torch.LongTensor([sample[0][-1]]))

        decoder_input = decoder_input.cuda()
        sequence, pre_scores, fin_scores, ave_scores, decoder_hidden, encoder_outputs = sample
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

        # choose topk
        topk = decoder_output.data.topk(beam_size)
        for k in range(beam_size):
            topk_prob = topk[0][0][k]
            topk_index = int(topk[1][0][k])
            pre_scores += topk_prob
            fin_scores = pre_scores - (k - 1 ) * 0.5
            samples.append([sequence+[topk_index], pre_scores, fin_scores, ave_scores, decoder_hidden, encoder_outputs])
        return samples

