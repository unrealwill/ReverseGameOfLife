import torch as th
import numpy as np
import os

#We use a finite world variant of game of life where outside border are "0" valued cells
#The size of the world is given by the shape of the state
def computeNextState( state ):
    weight = th.ones(1,1,3,3,dtype=th.float32,device=state.device)
    nbneighbors = th.nn.functional.conv2d( state, weight,padding="same") - state
    
    nbneighbors = th.round(nbneighbors).to(th.int32)
    #above rounding and casting to int is not necessary (and can be skipped) if state values are floating 0 or floating 1

    nextstate = (nbneighbors == 2)*state + (nbneighbors==3)
    return nextstate


#Very simple architecture : convolutional resnet variant
class PreviousStateModel(th.nn.Module):
    def __init__(self , dim,depth,kernelsize):
        super(PreviousStateModel, self).__init__()
        
        self.dim = dim
        self.kernelsize = kernelsize
        self.conv1 = th.nn.Conv2d(2,dim,self.kernelsize,padding="same")
        self.l = []
        self.lnorm = []
        self.depth = depth
        
        #TODO: better initialization for resnet layer so that they map to Id initially
        for i in range(self.depth):
            self.lnorm.append( th.nn.InstanceNorm2d(dim))
            self.l.append( th.nn.Conv2d(self.dim,dim,self.kernelsize,padding="same"))
            self.l.append( th.nn.Conv2d(dim,dim,self.kernelsize,padding="same"))
        self.convout = th.nn.Conv2d(dim,1,1,padding="same") 

        self.layers = th.nn.ModuleList(self.l) 
        self.layersnorm = th.nn.ModuleList(self.lnorm)              

    #model that takes a target state, and noisy state starting point and output predicted previous state
    #like we do in diffusion models because one state can have multiple different previous state
    def forward(self, tgt,noisystate):
        #we can eventually add other features like computeNextState(noisyState) and concat it into h
        h = self.conv1(th.concat([tgt,noisystate],dim=1))
        #in this resnet variant the non-linearity is not on the skip-path
        for i in range(self.depth):
            h = h + th.nn.functional.relu( self.l[2*i]( th.nn.functional.relu(self.l[2*i+1](self.lnorm[i](h) )) ))
        out = self.convout(h)
        return out


def sample( model, target , nbiter,initstate):
    state = initstate
    for i in range(nbiter):
        #we use linear noise schedule this can be improved (see diffusion litterature) 
        noiselvl = 0.5 - ((float)(i) / nbiter )*0.5
        #nextstate = computeNextState(state)
        #noiselvl = th.rand((bs,1,1,1),device=state.device) * 0.1
        #we flip the bit if a random number is smaller than the noiselvl
        noise = th.rand_like(initstate) < noiselvl
        noisystate = 0.5* ( (1-2*state) * (2*noise-1) +1 )
        #we use greedy sampling to go from logits to discrete state
        #alternatively we could sample with probability given by sigmoid(logits) but when there is a high number of bits it gets hard to have no error due to sampling
        #so for the last rounds (aka to finish) greedy deterministic action should be preferred 
        state = (model( target,noisystate) > 0.0).to(th.float32)
        nexstate = computeNextState(state)
        err = th.sum( abs(nexstate-target))
        print("iter " + str(i))
        print("err")
        print(err)
        if err == 0:
            print("Success! Antecedent found!")
            break

    return state



def train(n=10,bs=32,device="cuda",nbiter=100000):
    print("Training game of life backward model")

    model = PreviousStateModel(dim=256,depth=10,kernelsize=5).to(device=device)

    optimizer = th.optim.Adam( model.parameters(),lr=1e-4)

    for i in range (nbiter):
        with th.no_grad():
            state = th.randint(0,2,(bs,1,n,n)).to(th.float32).to(device=device)
            nextstate = computeNextState(state)
            rnd = th.rand((bs,1,1,1),device=state.device)
            #we can use various noise schedule (see various diffusion litterature like [2206.00364] )
            #when noiselvl is maximum the sample should not be discernable from pure noise
            #we square the rnd so that noisy samples are biased towards low noise (aka almost solved) to help learn to finish the solving
            noiselvl = rnd*rnd * 0.5
            #we flip the bit if a random number is smaller than the noiselvl
            noise = th.rand((bs,1,n,n),device=state.device) < noiselvl
            noisystate = 0.5* ( (1-2*state) * (2*noise-1) +1 )

        optimizer.zero_grad()
        #the model does not take as input the noise level, and will have to estimate it internally 
        logits = model( nextstate, noisystate )

        loss = th.nn.functional.binary_cross_entropy_with_logits( logits, state)
        #pred = (logits > 0).to(th.float32)
        #print( pred )
        loss.backward()
        if( i % 100 == 0):
            print("i : " + str(i) + " / " + str(nbiter))
            print("loss : " + str(loss))

        optimizer.step()
    
    th.save(model.state_dict(),"prevstatemodel.torch")

    

def predict(n=10,bs=1,device="cuda"):
    model = PreviousStateModel(dim=256,depth=10,kernelsize=5).to(device=device)
    model.load_state_dict(th.load("prevstatemodel.torch", weights_only=True))

    while True:
        state = th.randint(0,2,(bs,1,n,n)).to(th.float32).to(device=device)
        nextstate = computeNextState(state)

        initstate = th.randint(0,2,(bs,1,n,n)).to(th.float32).to(device=device)
        prevstate = sample(model,nextstate,100,initstate)
        nextprevstate = computeNextState(prevstate)
        print("state")
        print(state)
        print("nextstate")
        print(nextstate)
        print("prevstate")
        print(prevstate)
        print("nextprevstate")
        print(nextprevstate)
        input()



if __name__ == "__main__":
    n = 10
    device = "cuda" if th.cuda.is_available() else "cpu"
    nbiter= 100000  
    #if we want to evolve game of life
    bs = 1
    '''
    state = th.randint(0,2,(bs,1,n,n)).to(th.float32).to(device=device)
    for i in range(10):
        state = computeNextState(state)
        print("state")
        print(state)
    '''

    if( os.path.exists("prevstatemodel.torch") == False):
        train(n,32,device,nbiter)
    predict(n,1,device)

    
