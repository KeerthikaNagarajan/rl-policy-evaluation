# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.
## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
### States
The environment has 7 states:

* Two Terminal States: G: The goal state & H: A hole state.
* Five Transition states / Non-terminal States including S: The starting state.
### Actions
The agent can take two actions:

* R: Move right.
* L: Move left.
### Transition Probabilities
The transition probabilities for each action are as follows:

* 50% chance that the agent moves in the intended direction.
* 33.33% chance that the agent stays in its current state.
* 16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![267727933-ee9c6dcf-b579-4b1c-9663-47c8b17a08b4](https://github.com/KeerthikaNagarajan/rl-policy-evaluation/assets/93427089/3e2811ce-fd4d-444a-8188-6271548c52ab)
## POLICY EVALUATION FUNCTION
### Formula
![3](https://github.com/KeerthikaNagarajan/rl-policy-evaluation/assets/93427089/d91607b2-61b9-4f4f-b59b-c318a1b3e30c)
## Program:
### Policy Evaluation:
```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V

## Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

## Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

## Comparing policies based on state value function
### The state value function of the second policy V2 is greater than that of the first policy V1, so we conclude that the second policy is the best policy.

V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```
## OUTPUT:
### Policy 1:
![11](https://github.com/KeerthikaNagarajan/rl-policy-evaluation/assets/93427089/8b32f1d3-7b35-4169-a006-792550d2b37b)

### Policy 2:
![22](https://github.com/KeerthikaNagarajan/rl-policy-evaluation/assets/93427089/1f1bc7e5-5d59-4663-a00e-2b3c57a33498)

### Comparison:
![33](https://github.com/KeerthikaNagarajan/rl-policy-evaluation/assets/93427089/30a7555d-9924-4126-a14d-88c30ece9138)

### Conclusion:
![44](https://github.com/KeerthikaNagarajan/rl-policy-evaluation/assets/93427089/bdce79d0-87fb-4bef-9099-280a50b3a921)

## RESULT:

Thus, a Python program is developed to evaluate the given policy.

