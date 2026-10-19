# Plan 

Create a code which implements the best achievable solution and complies with the challenge rules.

# Context

The code base contains several baselines. The baseline closest to the real deployement is the Bayesian learning predictor with the epistemic reject option rule using the EM algorith. However, the baseline uses the true $p(x|y)$ which does not comply the challenge rules.

The goal is to implement a code following the challenge rules which creates a valid submission. The task is to achive the minimal achievalble score. The score will be evaluated on the public part of the test data, which will simulate the submition to the challenge system. The evaluation will have access to the test labels, however, only public part can be used.

The task is to create a stand-alone code with no dependencies on the this repo. The code base will contain:
- a script for installing on a fresh machine.
- a script that learns a predictor and creates a submission according to the challenge rules. 

Besides the code, the output should be a description of the implemented methods. The description analyze how the solution differs from the Bayesian+Epistemic+EM.

# Task

Implement the code described above to the folder `optimal_solution/`.



