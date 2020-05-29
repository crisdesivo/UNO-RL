''' Evaluating a Deep-Q Agent on UNO
'''

import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament


# Make environment
env = rlcard.make('uno', config={'seed': 0})

print(f"state shape: {env.state_shape}")

eval_env = rlcard.make('uno', config={'seed': 0})

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1


# Set a global seed
set_global_seed(0)

with tf.Session() as sess:

    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = DQNAgent(sess,
                     scope='dqn',
                     action_num=env.action_num,
                     replay_memory_size=20000,
                     replay_memory_init_size=memory_init_size,
                     train_every=train_every,
                     state_shape=env.state_shape,
                     mlp_layers=[60, 60, 60, 60, 60],
                     batch_size=512)

    saver = tf.train.Saver()
    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, "models/uno_dqn5/model")


    print(tournament(eval_env, 10000))
    
    
