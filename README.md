# kaggles
My attempts to do something on kaggle

## Playground s3e2 - Jan, 10, 2023
Task was to predict if person will have a stroke. Used DenseNetwork and rewised basics of Deep Learning. Beat benchmark and that's all. Didn't try to tune parameters because didn't wanted to. My best submission on public wart was 87% accuracy. 

The first approach was to use DenseNetwork straightforward but I failed to reach benchmark because it was unbalanced binary classification problem and my model lazily predicted that nobody would have stroke. Then, to fix this, I used WeightedRandomSampler and gave weight inversely proportional to the probability of stroke and monitored balanced accuracy while training.

## Digit Recognition - Jan, 11, 2023
This one is more interesting. Dataset contained 28x28 handwritten digits and task was to train model to distinguish them. Predicted it using CNN with Dropout and resulted in 99% accuracy.

Then, created small app, where you can draw digit and the model predicts which digit it is. Had some problem: 1) We trained model on digits written in grayscale, whule I use black-and-white in app, so had to retrain it (Noteworthy, this one gives 98% accuracy on test set); 2) Model is trained on small images while I draw them 400x400 in app and then resize them using linear interpolation. Moreover, not sure about thickness of digits, so model works poor on '6' and '9'. Perhaps, should try training it on more complicated dataset.
