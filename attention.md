# Attention

## 1. What is attention

Attention is one **component** of a network's architecture, there are two kinds of attention mechanism

a.	General Attention. Between input and output elements

b.	Self-attention. Within the input elements.



### 1.1 Attention in Seq2Seq

The standard seq2seq model is generally unable to accurately process long input sequences, since only the last hidden state of the encoder RNN is used as the context vector for the decoder.  the Attention Mechanism instead will use **all the hidden states of the input sequence** during the decoding process to  **pick out specific elements** from that sequence to produce the output 



## 2. Two types of attention

![]( https://github.com/ChrisWang10/NLP/raw/master/img/type1.png )











