import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output



class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(LSTM,self).__init__()
        self.rnn=nn.LSTM(input_size,hidden_size,bidirectional=False,batch_first=True)
        # self.linear=nn.Linear(hidden_size, output_size)
        self.dropout=nn.Dropout(p=0.2)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x hidden_size
        output=self.dropout(recurrent)
        # output = self.linear(output)  # batch_size x T x output_size
        return output