## Course
DATA130008.01 人工智能

## pbrain-Nuguri6
Ziqin LUO (18307130198), Yanda Li (18307130151)

#### How to compile the code
First, switch the path to the folder where the files are saved.
To compile the pruning agent, use the following command:
```
pyinstaller example.py pisqpipe.py --name pbrain-Nuguri6.exe --onefile
```
To compile the MCTS agent, use the following command:
```
pyinstaller example.py pisqpipe.py --name pbrain-MCTSSS.exe --onefile
```
#### What's our best agent? 
According to the combat results shown in our final report, we think alpha-beta pruning agent (pbrain-Nuguri6) is our best agent for Gomoku competition.
