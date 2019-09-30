from History import History


class Game(object):

    def __init__(self, agent, game_env):
        self.env = game_env
        self.agent = agent

    def simulate(self):
        env = self.env
        agent = self.agent

        observation = env.reset()

        history = History()
        sum_reward = 0
        done = False

        while not done:
            action, L1A, L2A, probs = agent.determine_action(observation)
            observation, reward, done, info = env.step(action)
            history.append(observation, L1A, L2A, probs, action, reward)
            sum_reward += reward

        return history, sum_reward
