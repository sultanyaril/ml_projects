# kaggles
My attempts to do something on kaggle

## Playground s3e2 - Jan 10, 2023
Task was to predict if person will have a stroke. Used DenseNetwork and rewised basics of Deep Learning. Beat benchmark and that's all. Didn't try to tune parameters because didn't wanted to. My best submission on public wart was 87% accuracy. 

The first approach was to use DenseNetwork straightforward but I failed to reach benchmark because it was unbalanced binary classification problem and my model lazily predicted that nobody would have stroke. Then, to fix this, I used WeightedRandomSampler and gave weight inversely proportional to the probability of stroke and monitored balanced accuracy while training.

## Digit Recognition - Jan 11, 2023
This one is more interesting. Dataset contained 28x28 handwritten digits and task was to train model to distinguish them. Predicted it using CNN with Dropout and resulted in 99% accuracy.

Then, created small app, where you can draw digit and the model predicts which digit it is. Had some problem: 1) We trained model on digits written in grayscale, whule I use black-and-white in app, so had to retrain it (Noteworthy, this one gives 98% accuracy on test set); 2) Model is trained on small images while I draw them 400x400 in app and then resize them using linear interpolation. Moreover, not sure about thickness of digits, so model works poor on '6' and '9'. Perhaps, should try training it on more complicated dataset.


## Disaster Recognition by Tweets - Jan 18, 2023
I wanted to do something about NLP because it seemed to be feasible for my PC. That was true but the problem itself was quite complicated. Most of the feature engineering was copied from someone's kaggle notebook and can be used in the future because of good plots. Struggled a few days trying to rewrite BERT used in notebook to torch but failed. I have to revisit this problem later and spend some time researching NLP solutions and examples.

After all, decided just to use sum of embeddings and use DenseNetwork then. Results are around 80% what is not bad but also not so good. 

Note: stop refering to Deep Learning models as to magic sticks. Have to do something with "Shallow Learning" models too.


## Intro to Game AI and Reinforcement Learning - Jan 21, 2023
The first course I complited on kaggle and the first certificate (yay! right before my bd).

The content is actually mid because I wrote only a few lines of code and they were very simple, but for basic understanding, it is enough, I guess. I liked RL very much and would like to do something more about it; perhaps, I should try watching lectures of London University which were recommended in the last lesson. 

