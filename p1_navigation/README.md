[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, I train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Run the notebook `Navigation.ipynb` to get started with training the own agent.

### Modeling Approach

* Overall we use a deep Q network (DQN) to train the agent using reinforcement learning.

* start with a simple DQN network and see if it is sufficient to solve the problem. The initial network I tried was a fully connected nn with sizes at each layer: start_size (input) -> 64 -> 128 -> action_size (out). But this model does not seem to be sufficiently rich. The score I got was moving around 10 while I think we should be able to move even higher.

* Then I tried a few other network structure. In the end I converge to the one I am submitting which is sufficient to get to score of ~15. The network is state_size -> 64 -> 128 -> 128 -> 64 -> action_size.

* Epsilon greedy algorithm is used, with esp started at 100%, decaying at 99.5% and floored at 1%.

* Replay buffer is used. We store upt o 10,000 memory and from that we randomly select experiences of batch size 64

* separately I have also tried to implement the following features to improve training:
  * [Double DQN](https://arxiv.org/abs/1509.06461) where use the latest Q net to pick the best next action, but use the target Q net to evaluate the action. This is controlled by the ddqn parameter when initializing the agent.
  * [Dueling](https://arxiv.org/abs/1511.06581) where we bifurcate the last layers of the network to separately predie the value function and the action specific adjustment on top. For the dueling we normalized the action output by the mean across actions. This is controlled by the dueling parameter when we initialize the agent.
  * [Prioried Experience Replay](https://arxiv.org/abs/1511.05952) where we prioritize the experience of higher TD difference. I also implemented the importance sampling weights as well. But the current implementation is extremely slow and more work is needed to make it useful. For now this feature is not used.
  
* I tried different combinations of training methods, ie, with and with out DDQN and Dueling respectively. This gives me 4 combinations. I ran these 4 models 10x each and plot out the resulting scores. To my surprise there is no observable pick up in the training speed across these few methodologies. it's possible that the network I used is too simple to observe material differences.

### Further Enhancements

We can try a few things to further improve the agent:
* fix the prioritized experience replay data structure and sampling logic to make it faster. We can reference the methodolgy [here](https://nn.labml.ai/rl/dqn/replay_buffer.html) and [here](https://github.com/facebookresearch/ReAgent/tree/main/reagent/replay_memory)
* We can go one step further and try to implement the Rainbow model [here](https://paperswithcode.com/method/rainbow-dqn).
* We can spend more time on finetuning hyper parameters. For the above runs, I basically juse used the same parameters I used in the exercise. It seems to work but we can definitely improve by more fine-tuning.
* We can try more complicated network structure, eg, more layers, more nodes, try CNN or other structure.


### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
