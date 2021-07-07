# ml-pacman
Machine learning agents that play a simplified version of pacman
The agents created were:
##### BasicAgentAA
Programatically designed agent
```
python3 busters.py -p BasicAgentAA -l layout.lay
```
##### AutomaticAgent
Agent built using a classification to predict pacman actions. The final model uses ibk(k=3)
```
python3 busters.py -p AutomaticAgent -l layout.lay
```
##### QLearningAgent
Agent built using reinforcement learning with the Q-Learning method
```
python3 busters.py -p QLearningAgent -l layout.lay
```
