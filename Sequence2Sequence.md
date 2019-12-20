# Seq2Seq

![]( https://github.com/ChrisWang10/NLP/raw/master/img/seq2sseq.png )



seq2seq is widely used in the field of **Machine Translation** , The most common architecture used to build a seq2seq is The **Encoder Decoder** architecture.

***Encoder***  read the input sequence and extract the information we need. Typically we use LSTM to achieve this. But it's worth to be noted that we discard the output of the Encoder and only preserve the internal states.

***Decoder*** is also built by LSTM. In training phase, we use the **actual output** of previous time step as the input of each time step to help to train faster. In inference phase, we use predict output instead.

![](https://github.com/ChrisWang10/NLP/raw/master/img/decoder.png)