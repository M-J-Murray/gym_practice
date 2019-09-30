from Game import Game
import gym


class Emulator(object):

    def __init__(self, game, agent, seed):
        self.agent = agent
        game_env = gym.make(game)
        game_env.seed(seed)
        self.game = Game(agent, game_env)

    def start(self):
        episode = 0
        running_reward = None

        while True:
            episode += 1
            history, sum_reward = self.game.simulate()
            self.agent.update_params(history)
            running_reward = sum_reward if running_reward is None else running_reward * 0.999 + sum_reward * 0.001
            if episode % 100 == 0:
                print "episode ", episode, " complete - average reward = ", running_reward
