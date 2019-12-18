# Word2Vec

## Intuition:

​	we need to find a vec to represent the words. First we may find one-hot encoder to represent words. But different words contains similar meaning should have close vector distance,then one-hot encoder form won't be acceptable.

## skip gram

We train a model that can predict the **possibility of eah word in our vocabulary being the nearby word**(we can define a nearby window, generally 3 or 5 is ok) given an input word. For example, 'susan and I have a dog'. given an input word 'a'. If nearby window is 2 and our vocabulary contains 10 words, then our model can get a vector which shape is (1, 10). 

