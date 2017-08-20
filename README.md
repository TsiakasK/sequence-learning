# sequence-learning
**Interactive Learning and Adaptation for the Sequence Learning task with NAO**
- Supervised Actor Critic with Policy Gradient Reinforcement Learning
- Dynamic User Modeling 
- EEG engagement monitoring using MUSE (Learning from Feedback)
- Online GUI Robot Learning (Learning from Guidance)

## Requirements
- Ubuntu 14.04 or later
- MUSE SDK for python http://developer.choosemuse.com/ 
- python pyliblo wrapper http://das.nasophon.de/pyliblo/
- NAOqi for python http://doc.aldebaran.com/2-5/dev/python/install_guide.html
After doing export, we need to do echo 'export PYTHONPATH={PYTHONPATH}:/home/username/pynaoqi/' >> ~./bashrc
- numpy, scipy, matplotlib
- TODO: create install.sh file for installing requirements

## Running instructions
- Run muse-io
muse-io --device Muse-XXXX --osc osc.udp://localhost:5000
- Run play.py 
(make sure the port number is the same in play.py and muse_pyliblo_server.py files -- TODO: create launch file to run these automatically)

*Note: for the purposes of the game, we have built a buzzer-like box with EASY(R) buttons for the user to respond, responses can be also recorder through keyboard* 

## output files
- MUSE output files
During the interaction, we collect MUSE data (1) when the robot announces the sequence and (2) when the user reponds

- Each line on the file starts with a character, each with a specific meaning:
h - receive horseshoe - status indicator values
eeg – EEG Data
a – Alpha relative 
b – Beta relative 
g – Gamma relative
d – Delta relative
t – Theta relative
Aa – Alpha absolute
Ab – Beta absolute
Ag – Gamma absolute
Ad – Delta absolute
At – Theta absolute
as – Alpha session score [Session score info]
bs – Beta session score
gs – Gamma session score
ds – Delta session score
ts – theta session score
c – concentration
Each line has four readings from sensors in left ear, left forehead, right forehead, right ear.

- Robot_#
This file records data from Muse when user is listening to the robot while it is announcing the sequence

- User_#
This file records data from Muse when user is responding by pressing the buttons

- logfile:
For each round the following details are recorded: 
Turn number, length of sequence, robot feedback, current score, success (1) / failure (-1), reaction time, completion time, sequence given by robot, sequence entered by user.
Reaction time: Time until user enters the first character in the sequence.
Completion time: Time until user completes the entire sequence.

- state_EEG:
In each round, the below details are recorded:
Length of the sequence, robot feedback type, previous score (users score from the previous turn/round), corresponding EEG filenames
Score is calculated by the formula:  result*difficulty_level, where result = [-1, 1] and difficulty_level = [1,2,3,4]

