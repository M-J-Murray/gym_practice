#!/home/michael/anaconda3/envs/AIGym/bin/python3.6
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
import torch.optim as optim
import torch.nn as nn
import torch
from Model import Model
from Worker import Worker
import gym

def main():
    episode = 0
    path = "/home/michael/dev/fyp/AIGym/MP-Conv-Pong/"
    mp.set_start_method('spawn')
    worker_count = 3#mp.cpu_count()
    learning_rate = 1e-3
    model = Model(2)
    criterion = nn.CrossEntropyLoss(reduce = False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if episode > 0:
        model.load_state_dict(torch.load(path+"Models/"+str(episode)))
        optimizer.load_state_dict(torch.load(path+"Optimizers/"+str(episode)))
    model.cuda()
    model.share_memory()

    envs = [gym.make("Pong-v0") for i in range(worker_count)]

    batch_save = 100
    best_score = None
    running_reward = None
    reward_queue = SimpleQueue()

    #Start workers
    workers = [Worker(envs[i], model, criterion, optimizer, reward_queue, str(i+1)) for i in range(worker_count)]
    [w.start() for w in workers]

    # Gather rewards
    while True:
        reward = reward_queue.get()
        if not isinstance(reward, float):
            print(reward)
        else:
            episode += 1
            if (episode % batch_save == 0):
                torch.save(model.state_dict(), path+"Models/"+str(episode))
                torch.save(optimizer.state_dict(), path+"Optimizers/"+str(episode))
                
            if best_score is None:
                best_score = reward
            elif reward > best_score:
                best_score = reward
            running_reward = reward if running_reward is None else running_reward * 0.99 + reward * 0.01
            if episode % 1 == 0:
                print("episode {:4.0f} complete - average reward = {:3.0f}, last score was = {:3.0f}, best score is = {:3.0f}".format(episode,
                                                                                                            running_reward,
                                                                                                            reward,
                                                                                                            best_score))

if __name__ == '__main__':
    main()
