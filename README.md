This is our fork of the B-Gap repository.

In order to run our update model we use the original instruction with the local path of our best model:

`python3 -W ignore experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/dqn.json --test --episodes=25 --name-from-config --recover-from=/Users/Lakhsh/School/geom/final/b-gap/rl-agents/scripts/checkpoint-999.tar --no-display`