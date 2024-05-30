
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()

paths_list = filedialog.askopenfilenames(parent=root, title='Choose some txt files')
num_files = len(paths_list)


episode = []
reward = []
reward_avg = []


for i in range (0, num_files):
    
    episode.append([])
    reward.append([])

    f = open(paths_list[i], 'r')
    lines = f.readlines()

    j = 0

    for line in lines:
        values_list = line.split(",")                   # values_list = ["ep:40", "reward:22.25", ...]
        
        episode[i].append(int(values_list[0].split(":")[1])) #one array for each episode and reward
        reward[i].append(float(values_list[1].split(":")[1]))

        if i == 0:
            reward_avg.append(float(values_list[1].split(":")[1])) #one array for all episodes and rewards to do average
        else:
            reward_avg[j] += float(values_list[1].split(":")[1])
        j += 1


    f.close()


# Compute average
for i in range(0, len(reward_avg)):
    reward_avg[i] = reward_avg[i]/num_files


for i in range (0, num_files):
    plt.plot(episode[i], reward[i], label = str(i), color="#d4d4d4")
    #plt.scatter(episode[i], reward[i], label = str(i), color="#d4d4d4")

plt.plot(episode[0], reward_avg, color="blue", linewidth = '3')


font1 = {'color':'black','size':15}
font2 = {'color':'blue','size':20}
plt.title("Average reward on " + str(num_files) + " runs", fontdict = font2)
plt.xlabel("Episode", loc="right", fontdict = font1)
plt.ylabel("Reward", loc="center", fontdict = font1)

#plt.legend(loc='upper left')
plt.show()








