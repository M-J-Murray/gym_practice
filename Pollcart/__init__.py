from BPAgent import BPAgent
from Emulator import Emulator

#CartPole-v0
#Pong-v0
game = 'CartPole-v0'
seed = 50
height = 160
dimensionality = 4
actions = 2
learning_rate = 1e-4
agent = BPAgent(height, dimensionality, actions, learning_rate, seed)

emulator = Emulator(game, agent, seed)

emulator.start()
