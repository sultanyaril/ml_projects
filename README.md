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

## [RL course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) - Feb 23, 2023
![](https://github.com/sultanyaril/kaggles/blob/main/david%20silver's%20rl%20course/easy21_illustration.jpg)
As I suggested, started watching great course from University College London. Unfortunately, had to make a little pause because of the beggining of a new term. However, I watched 4 lectures of the course just before that, and only month later started doing assignment.

It took time to finish it, but I memorized concepts we met on the lectures and done a great job. After finishing assignment, I wanted to compare my work to the results of other students and find out that there was a little done dilligently: some of the had errors as value function being more than 1 (while it cannot be more than reward), too big MSE, not checking behavior of TD function on lambda = 1, etc. I believe my results are correct. However, I have to admit that copied answers to the last 5 open questions from [here](https://github.com/kvfrans/Easy21-RL) 🙈. Other parts are done by me and sometimes code is too sloppy, lacks routine and readability, but the most important thing is results - pictures.

Now, I unlocked next lectures for me, and will spend time to watch them parallel to my study at University. Also, attached notes of the lectures.

## Mar 30, 2023

Took some time to complete the course because of university. Managed to complete all lectures and write the final paper, scored 28/40 what is not bad for a person who wasn't preparing (but used notes when writing). Now, I want to try to apply this knowledge in creating some reinforcement learning model of soccer. Hope, it will work. Hope, it will not take half a year.

## Flappy Bird RL - Apr 27, 2023
![](https://github.com/sultanyaril/kaggles/blob/main/flappy%20bird%20rl/gifs/env_simple.gif)
![](https://github.com/sultanyaril/kaggles/blob/main/flappy%20bird%20rl/gifs/env_12.gif)
<img src="https://github.com/sultanyaril/kaggles/blob/main/flappy%20bird%20rl/gifs/big_score.png" width="288" height="512">

My first RL project! After completion of David Silver's course, I have decided to apply my knowledge and found 'gym' where flappy bird environment was present. I thought that it might be fun to train an agent who can play this game. Results are not impressive but it works.

I [have](https://github.com/markub3327/flappy-bird-gymnasium) [used](https://github.com/yenchenlin/DeepLearningFlappyBird) a [lot](https://github.com/Talendar/flappy-bird-gym) of [different](https://github.com/samboylan11/flappy-bird-gym) people's gits and very thankful for their repositories. 

What I have done contains: models trained on 2 features (vertical and horizontal distance to the next bar), 12 features (position of bird and position of next 3 bars) and RGB features (whole scene). Unfortunately, I couldn't train last one because it uses pygame and my PC is not that good to render it fast. Then, I have trained model on 8 million steps (4 million with EpsilonGreedy, where epsilon goes down from 0.1 to 0.001, and 4 million on the resulted policy). The last model survives for about 200 scores in mean, but covariance is still too big and it can die anytime. Also, there you can find two NNs trained on 1 millions steps to compare how good 12 features is comparing it to 2. Finally, I could have worked more to reduce variance and train it more to get more fascinating results but it would take too much time.

Now, I have introduced myself to 'gym' environment and I really want to try to write everything all by myself. Moreover, I want to try to train model which plays with itself.

## Tennis RL - May 19, 2023

HERE NEED PICTURES!!!

This one is 100% made by me and every line belongs to me (to the extent that code can belong to someone). Actually, learned a lot of interesting things and faced different problems while implementing code. 

This code includes Pong environment written on gym. While writing the game, I faced problem that ball stuck inside of the paddle but it was solved by allowing it to bounce only once (checks if ball's velocity is towards a paddle). Pong environment is very simple: ball have an elastic collision with paddle with conversation of energy and there's no randomness and velocity do not increase. Rewards system is a little bit complicated for me to custom it when training: 5 points for hit of left paddle and 1 point for scoring of left paddle, and negative values for the same actions of right paddle. 

Also, I'm happy that I made it as an executable library and it can be run easily by writing:
```
python3 -m tennis_gym [human | robot | god] [human | robot | god]
```
'human' is a user which can move paddle by 'W' and 'S' or 'Up' and 'Down' depending on the side; 'robot' is an agent which works using the model I trained; 'god' is never-losing agent which moves after the ball.

Now, about model. If writing of the environment took me day or two, training model had 2 weeks of my life. I used DQNAgent and implemented it as a class with all necessary routine. One thing I knew when implementing of the project: there is double model (model and target_model) system which helps with convergence (proved by paper). Also, I knew how much torch is faster than keras. Let's discuss it a little more: first my model was made using keras and it worked really slow, 10000 episodes would have taken 2 days to train. But, it is not an only problem: memory leaks (!) when predicting actions. So, it was decided to use torch instead and I'm happy that I did it because it takes only 2.5 hours to train on 10000 episodes.

The end result was made by training DQNAgent on playing with itself for four times on 10000 episodes with Lineary Annealing e-greedy Q-policy. It learned on 8 features: ball's x position, ball's y position, ball's x velocity, ball's y velocity, player1's x position, player1's y position, player2's x position, player2's y position. There were 3 actions: move up, move down, stay. About rewards, agent is rewarded 100000 for hitting the ball and penalized 20000 for conceding a goal. Values chosen to be big for better propogation (need to checked if it's true). As a result, on 40000 episodes I've gotten a model which draws all 10000 episodes played with god, what I believe is the best result that could have been produced (code is in test_agent.py).

One interesting thing I implemented: as environment is symmetric, I trained only one Neural Network by mirroring environment to calculate right paddle's action, so it made training much faster by twicing the number of steps that it can learn on.

Things to try (maybe):
1. Try to train it using Convolutional NN on the whole field;
2. Check how the average number of steps changes while training because I saw that number of drawn games can dramatically lower in just 500 episodes (from 80 to 20), but maybe average number of steps is increasing;
3. Use different reward. I learned that reward is the most crucial part of RL model. Perhaps using 'vanilla' reward of 1 for scoring and penalty of 1 for conceding can provide a result but it will require more steps.

My next goal was to implement football environment (2D for now) and try to teach footballers to play with each other to score goals. Now, I see that 'sparse' reward of 1 for scoring and 1 penalty for conceding can be a problem. Moreover, I struggled a lot with some basic concepts when implementing solution for Pong so I need to see more examples of RL implementations and models. I believe that before tackling football, I need to read Maxim Lapan's book on RL. So, it will be my next destination.


