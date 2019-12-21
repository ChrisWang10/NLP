# Attention

## 1. What is attention

Attention is one **component** of a network's architecture, there are two kinds of attention mechanism

a.	General Attention. Between input and output elements

b.	Self-attention. Within the input elements.



### 1.1 Attention in Seq2Seq

The standard seq2seq model is generally unable to accurately process long input sequences, since only the last hidden state of the encoder RNN is used as the context vector for the decoder.  the Attention Mechanism instead will use **all the hidden states of the input sequence** during the decoding process to  **pick out specific elements** from that sequence to produce the output. Attention Mechanism allows the decoder to attend to different parts of the source sentence at each step of the output generation. Instead of encoding the input sequence into a **single fixed context vector**, we let the model learn **how to generate a context vector** for each output time step.  



## 2. Two types of attention

![]( https://github.com/ChrisWang10/NLP/raw/master/img/type1.png )





### 2.1	produce encoder hidden states

### 2.2	calculating alignment score

The alignment score evaluate how well each encoded input matches the current output of the decoder.

![](https://github.com/ChrisWang10/NLP/raw/master/img/alignment-score.png)

### 2.3 Softmax the alignment score

 we can then apply a softmax on this vector to obtain the attention weights. 

### 2.4 Context  vector

 After computing the attention weights in the previous step, we can now generate the context vector by doing an element-wise multiplication of the attention weights with the encoder outputs. 

### 2.5 Decode the ouput

 The context vector we produced will then be concatenated with the previous decoder output. It is then fed into the decoder RNN cell to produce a new hidden state.







