# DisasterTweets
Deep Learning models to identify whether tweets contain real disasters or not.

# Preprocessing
Data format before preprocessing:

  id keyword location                                               text  target
0   1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...       1
1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada       1
2   5     NaN      NaN  All residents asked to 'shelter in place' are ...       1
3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...       1
4   7     NaN      NaN  Just got sent this photo from Ruby #Alaska as ...       1
5   8     NaN      NaN  #RockyFire Update => California Hwy. 20 closed...       1
6  10     NaN      NaN  #flood #disaster Heavy rain causes flash flood...       1
7  13     NaN      NaN  I'm on top of the hill and I can see a fire in...       1
8  14     NaN      NaN  There's an emergency evacuation happening now ...       1
9  15     NaN      NaN  I'm afraid that the tornado is coming to our a...       1

Steps for preprocessing:
1. removed id, keyword, and location columns
2. removed URLs from text
3. made all the text lowercase
4. removed emojis, punctuation, symbols

Processed data looks like this:
                                                   text  target
691   shouout to kasad1lla cause her vocals are blaz...       0
4086  hail mary full of grace the lord is with thee ...       0
7146  brian shaw jj hickson kenneth faried trying to...       0
5486  yet another company trying to censor the inter...       0
2578  engineermatarai ate mataas kc ratingbut they d...       0
2212             photo forrestmankins colorado camping        0
266   us national park services tonto national fores...       0
4778  kabwandi breaking news unconfirmed i just hear...       0
6497  damn theres really no mlk center that hasnt su...       0
4283  day 3 without my phone and due to my slow shit...       0

## Split into training and testing texts, and training and testing labels.
- X_train (4906,) of training text
- Y_train (4906,) of training labels
- X_test (1636,) of testing text
- Y_test (1636,) of testing labels

For LSTM Model, I also tokenized the inputs in the main function if you decide you need them.
