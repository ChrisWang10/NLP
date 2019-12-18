# Word2Vec

## Intuition:

​	we need to find a vec to represent the words. First we may find one-hot encoder to represent words. But different words contains similar meaning should have close vector distance,then one-hot encoder form won't be acceptable.

## skip gram

We train a model that can predict the **possibility of eah word in our vocabulary being the nearby word**(we can define a nearby window, generally 3 or 5 is ok) given an input word. For example, 'susan and I have a dog'. given an input word 'a'. If nearby window is 2 and our vocabulary contains 10 words, then our model can get a vector which shape is (1, 10). 

![]( https://github.com/ChrisWang10/NLP/raw/master/img/skip-gram.png )

**The hidden layer paramters are what we need. **

we can see  If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. So, if two words have similar contexts, then our network is motivated to learn similar word vectors for these two words! 



## problem

this model contains a huge number of weights,  For our example with 300 features and a vocab of 10,000 words, that’s 3M weights in the hidden layer and output layer each .

**subsampling frequent words**

![]( https://github.com/ChrisWang10/NLP/raw/master/img/training-sample.png )

We can notice that 'the' appears in the context of pretty much every word. And we only need use a fraction of 'the' then we can get a good representation for it.

One way is to delete the frequent words with a possibility which is obtained by the frequency of word appear in the sentence.

wi is the word and z(wi) is the frequency.Then the possibility of keeping the word is 

![](https://github.com/ChrisWang10/NLP/raw/master/img/possiblility.png)

**Negative Sampling**

