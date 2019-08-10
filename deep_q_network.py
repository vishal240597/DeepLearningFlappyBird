#!/usr/bin/env python
#updating the code
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()


import gym
import numpy as np
import random
from IPython.display import clear_output

# Init Taxi-V2 Env
env = gym.make("Taxi-v2").env

# Init arbitary values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1


all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    # Init Vars
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            # Check the action space
            action = env.action_space.sample()
        else:
            # Check the learned values
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Update the new value
        new_value = (1 - alpha) * old_value + alpha * \
            (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: {i}")

print("Training finished.\n")

env = gym.make("Taxi-v2").env

env.s = 328

epochs = 0
penalties, reward = 0, 0

frames = []

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    }
    )

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

print_frames(frames)

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE])) 

gamma = 0.75
# learning parameter 
initial_state = 1

# Determines the available actions for a given state 
def available_actions(state): 
	current_state_row = M[state, ] 
	available_action = np.where(current_state_row >= 0)[1] 
	return available_action 

available_action = available_actions(initial_state) 

# Chooses one of the available actions at random 
def sample_next_action(available_actions_range): 
	next_action = int(np.random.choice(available_action, 1)) 
	return next_action 


action = sample_next_action(available_action) 

def update(current_state, action, gamma): 

max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1] 
if max_index.shape[0] > 1: 
	max_index = int(np.random.choice(max_index, size = 1)) 
else: 
	max_index = int(max_index) 
max_value = Q[action, max_index] 
Q[current_state, action] = M[current_state, action] + gamma * max_value 
if (np.max(Q) > 0): 
	return(np.sum(Q / np.max(Q)*100)) 
else: 
	return (0) 
# Updates the Q-Matrix according to the path chosen 

update(initial_state, action, gamma) 

scores = [] 
for i in range(1000): 
	current_state = np.random.randint(0, int(Q.shape[0])) 
	available_action = available_actions(current_state) 
	action = sample_next_action(available_action) 
	score = update(current_state, action, gamma) 
	scores.append(score) 

# print("Trained Q matrix:") 
# print(Q / np.max(Q)*100) 
# You can uncomment the above two lines to view the trained Q matrix 

# Testing 
current_state = 0
steps = [current_state] 

while current_state != 10: 

	next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1] 
	if next_step_index.shape[0] > 1: 
		next_step_index = int(np.random.choice(next_step_index, size = 1)) 
	else: 
		next_step_index = int(next_step_index) 
	steps.append(next_step_index) 
	current_state = next_step_index 

print("Most efficient path:") 
print(steps) 

pl.plot(scores) 
pl.xlabel('No of iterations') 
pl.ylabel('Reward gained') 
pl.show() 
# Defining the locations of the police and the drug traces 
police = [2, 4, 5] 
drug_traces = [3, 8, 9] 

G = nx.Graph() 
G.add_edges_from(edges) 
mapping = {0:'0 - Detective', 1:'1', 2:'2 - Police', 3:'3 - Drug traces', 
		4:'4 - Police', 5:'5 - Police', 6:'6', 7:'7', 8:'Drug traces', 
		9:'9 - Drug traces', 10:'10 - Drug racket location'} 

H = nx.relabel_nodes(G, mapping) 
pos = nx.spring_layout(H) 
nx.draw_networkx_nodes(H, pos, node_size =[200, 200, 200, 200, 200, 200, 200, 200]) 
nx.draw_networkx_edges(H, pos) 
nx.draw_networkx_labels(H, pos) 
pl.show() 

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE])) 
env_police = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE])) 
env_drugs = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE])) 
initial_state = 1

# Same as above 
def available_actions(state): 
	current_state_row = M[state, ] 
	av_action = np.where(current_state_row >= 0)[1] 
	return av_action 

# Same as above 
def sample_next_action(available_actions_range): 
	next_action = int(np.random.choice(available_action, 1)) 
	return next_action 

# Exploring the environment 
def collect_environmental_data(action): 
	found = [] 
	if action in police: 
		found.append('p') 
	if action in drug_traces: 
		found.append('d') 
	return (found) 


available_action = available_actions(initial_state) 
action = sample_next_action(available_action) 

def update(current_state, action, gamma): 
max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1] 
if max_index.shape[0] > 1: 
	max_index = int(np.random.choice(max_index, size = 1)) 
else: 
	max_index = int(max_index) 
max_value = Q[action, max_index] 
Q[current_state, action] = M[current_state, action] + gamma * max_value 
environment = collect_environmental_data(action) 
if 'p' in environment: 
	env_police[current_state, action] += 1
if 'd' in environment: 
	env_drugs[current_state, action] += 1
if (np.max(Q) > 0): 
	return(np.sum(Q / np.max(Q)*100)) 
else: 
	return (0) 
# Same as above 
update(initial_state, action, gamma) 

def available_actions_with_env_help(state): 
	current_state_row = M[state, ] 
	av_action = np.where(current_state_row >= 0)[1] 

	# if there are multiple routes, dis-favor anything negative 
	env_pos_row = env_matrix_snap[state, av_action] 

	if (np.sum(env_pos_row < 0)): 
		# can we remove the negative directions from av_act? 
		temp_av_action = av_action[np.array(env_pos_row)[0]>= 0] 
		if len(temp_av_action) > 0: 
			av_action = temp_av_action 
	return av_action 
# Determines the available actions according to the environment 
scores = [] 
for i in range(1000): 
	current_state = np.random.randint(0, int(Q.shape[0])) 
	available_action = available_actions_with_env_help(current_state) 
	action = sample_next_action(available_action) 
	score = update(current_state, action, gamma) 
	scores.append(score) 

