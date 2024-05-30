
import gym
import numpy as np
import random
import datetime
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow.keras.backend as K
from collections import deque



# Instantiate environment

env = gym.make("CartPole-v1", render_mode = "human")
env.action_space.seed(31)

num_observations = len(env.observation_space.sample())
num_actions = env.action_space.n


# To log stuff:
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logs_dir = "./logs/graph" + current_time + ".txt"


f = open(logs_dir, "w", encoding="utf-8")
f.write("")
f.close()



# past experience
EXP_MAX_SIZE = 2000 # Max batch size of past experience
BATCH_SIZE = 600
experience = deque([], EXP_MAX_SIZE) # Past experience arranged as a queue


EPS_MAX = 100
EPS_MIN = 5
EPS_DECAY = 1.5          # how much epsilon is reduced every 'episode_batch' episodes
MAX_EPS_DECAY = 4
GAMMA = 0.99 # discount factor
LR = 0.012
LR_MIN = 0.0006
c_reward = 0
checkpoint_path = './checkpoints19/cp.ckpt'

RANDOM_EPS = 20


# NN architecture
model = Sequential()
model.add(Dense(256, input_shape=(num_observations,), activation='relu'))       # 4 observations
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.05))
model.add(Dense(2, activation='linear'))                        # 2 outputs
opt = tf.keras.optimizers.SGD(learning_rate=LR)
model.compile(optimizer=opt, loss='mse')




# initialize environment and get related information and observation
obs, info = env.reset(seed=17)

episode = 1
epsilon = EPS_MAX
episode_batch = 5      # = how often training, epsilon reduction and logs are done


# Logs
avg_reward = 0
current_lr = LR
loss = 0
val_loss = 0



while (episode <= 300):
        act = 0
        prediction = model.predict_on_batch(tf.constant([[*obs]]))[0]

        rv = random.randint(1,100)

        if rv < epsilon:
                act = env.action_space.sample() # pick random action

        else:
                if (prediction[0] >= prediction[1]):
                        act = 0
                else:
                        act = 1

        obs_next, reward, terminated, truncated, info = env.step(act) # execute action and collect corresponding info

        if not terminated:
                c_reward += reward

        prediction_next = model.predict_on_batch(tf.constant([[*obs_next]]))[0]
        reward_next = max(prediction_next[0], prediction_next[1])

        if terminated:
                reward = -50
        else:
                reward = 1 + GAMMA*reward_next

        if (act == 0):
                reward0 = reward
                reward1 = prediction[1]
        else:
                reward0 = prediction[0]
                reward1 = reward


        experience.append([*obs, reward0, reward1])    

        if len(experience)>= EXP_MAX_SIZE:
                experience.popleft()


                
        obs = obs_next # update current state


        if terminated or truncated: # if episode ended



                if episode >= RANDOM_EPS and episode % episode_batch == 0:  # first episodes are all random
                        epsilon -= EPS_DECAY
                        if epsilon <= EPS_MIN:
                                epsilon = EPS_MIN
                        EPS_DECAY = min(MAX_EPS_DECAY, EPS_DECAY + 0.005)        # epsilon deceleration


                avg_reward += c_reward
                
                if episode % episode_batch == 0:

                        if len(experience) >= BATCH_SIZE:

                                batch = random.sample(experience, BATCH_SIZE)

                                #prepare data
                                dataset = np.array(batch)
                                X = dataset[: , : num_observations]     # X is all the dataset, for each element we take the first 4 values (observations, 0 to 3)
                                Y = dataset[: , num_observations : num_observations + num_actions]    # Y is all the dataset, for each element we take values 4 and 5 (reward0 e reward1)

                                #train network
                                result = model.fit(tf.constant(X),
                                                   tf.constant(Y),
                                                   validation_split = 0.1,
                                                   epochs = 2
                                                   )

                                loss = result.history["loss"][-1]               
                                val_loss = result.history["val_loss"][-1]

                                current_lr = K.get_value(model.optimizer.lr)
                                print ("Learning rate:", current_lr)

                                
                                # Decrease learning rate:
                                if episode > 60:
                                        new_LR = current_lr - 0.0002
                                        new_LR = max(new_LR, LR_MIN)
                                        K.set_value(model.optimizer.lr, new_LR)
 

                        avg_reward /= episode_batch
                        print("Average reward (", episode-episode_batch, "-", episode,") =", avg_reward)

                        f = open(logs_dir, "a", encoding="utf-8")
                        string_to_write = "ep:" + str(episode) + ", reward:" + str(avg_reward)
                        string_to_write += ", epsilon:" + format(epsilon, '.2f') + ", lr:" + format(current_lr, '.4f')
                        string_to_write += ", loss:" + format(loss, '.4f') + ", val_loss:" + format(val_loss, '.4f') + "\n"
  
                        f.write(string_to_write)
                        f.close()
                        
                        avg_reward = 0



                # print debug information
                print("----------------------------------episode", episode)
                print("reward =", c_reward)
                print("epsilon =", epsilon)
                episode += 1
                obs, info = env.reset(seed = rv)
                c_reward = 0
env.close()
