# ReverseGameOfLife
Reverse Game Of Life Using Diffusion Model

Try to predict the previous state of a game of life state using a diffusion model.

Saw AlphaPhoenix video https://www.youtube.com/watch?v=g8pjrVbdafY for more explanation about the problem.

Here we are not trying to find all antecedents, but any one.
We train a diffusion model on a discrete state space : 
- We take a state, compute the next state
- We corrupt the state with random bit flip depending on the noise level
- And we train a convolutional residual neural network to predict model( nextstate, noisystate) -> state

Once the model is trained, to sample we follow the diffusion methodology :
-we start from random noise and target state
-In a loop with decreasing noise levels :
-predict the denoised state from target state and current noisy state with the model
-we add noise by randomly flipping some bits

Requirement :
torch 

To run :
```python3 gameoflife.py```

On the first run it will train a diffusion model, and save it for subsequent run. (The model is 126M so not sharing it on github).

It will generate a random state, compute the next state, and then reverse this next state so we know there is at least a solution (not a garden of eden).
Have not computed success statistics for but probably > 80% for 10x10 worlds with less than 100 iterations.

example output prediction :
```
iter 61
err
tensor(6., device='cuda:0')
iter 62
err
tensor(4., device='cuda:0')
iter 63
err
tensor(12., device='cuda:0')
iter 64
err
tensor(0., device='cuda:0')
Success! Antecedent found!
state
tensor([[[[1., 1., 0., 1., 0., 1., 1., 1., 1., 1.],
          [0., 1., 0., 0., 0., 0., 0., 0., 1., 1.],
          [1., 0., 1., 0., 1., 0., 1., 1., 1., 0.],
          [0., 1., 0., 1., 0., 0., 0., 0., 1., 1.],
          [1., 1., 0., 1., 1., 1., 0., 0., 0., 1.],
          [1., 1., 0., 0., 0., 0., 0., 1., 1., 0.],
          [0., 1., 1., 1., 0., 1., 0., 1., 1., 0.],
          [0., 0., 0., 0., 1., 1., 0., 1., 1., 0.],
          [1., 0., 0., 1., 1., 1., 0., 1., 0., 1.],
          [1., 1., 1., 1., 1., 1., 0., 1., 0., 1.]]]], device='cuda:0')
nextstate
tensor([[[[1., 1., 1., 0., 0., 0., 1., 1., 0., 1.],
          [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
          [1., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
          [0., 0., 0., 1., 1., 0., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
          [1., 1., 1., 1., 0., 1., 0., 0., 0., 1.],
          [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
          [1., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
          [1., 1., 1., 0., 0., 1., 0., 0., 0., 0.]]]], device='cuda:0')
prevstate
tensor([[[[1., 1., 1., 0., 0., 0., 0., 1., 1., 1.],
          [0., 1., 0., 0., 0., 1., 0., 1., 1., 1.],
          [1., 0., 0., 1., 1., 0., 0., 1., 1., 0.],
          [1., 1., 0., 0., 1., 1., 1., 0., 1., 1.],
          [1., 1., 1., 1., 1., 0., 0., 1., 0., 1.],
          [1., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
          [0., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
          [0., 0., 0., 0., 1., 1., 0., 1., 1., 1.],
          [1., 0., 0., 1., 1., 0., 0., 1., 0., 1.],
          [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.]]]], device='cuda:0')
nextprevstate
tensor([[[[1., 1., 1., 0., 0., 0., 1., 1., 0., 1.],
          [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
          [1., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
          [0., 0., 0., 1., 1., 0., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
          [1., 1., 1., 1., 0., 1., 0., 0., 0., 1.],
          [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
          [1., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
          [1., 1., 1., 0., 0., 1., 0., 0., 0., 0.]]]], device='cuda:0')
```
