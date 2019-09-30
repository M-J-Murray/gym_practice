import numpy as np

# pre-processes the frames returned by the game, so that they are suitable for the network
def prepro(frame):
    frame = frame[32:198, 16:144]  # crop
    frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2] # greyscale
    frame = frame[::4, ::2]
    return frame.reshape(1, 42, 64, 1).astype("uint8")


def discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Randomly selects an action from the supplied distribution f
def choose_action(f):
    hot = np.zeros_like(f, dtype="uint8")
    th = np.random.uniform(0, 1)
    run_sum = 0
    i = 0
    for i in range(f.size):
        run_sum += f[0, i]
        if th < run_sum:
            break
    hot[0, i] = 1
    return hot


# trains a model using the training dataset by randomly sub-sampling batches based on the batch_size.
# Note how the gradient is kept from every batch and then used to adjust the network weights
def train(sess, model, history):
    observations = np.concatenate(history["observations"])
    actions = np.concatenate(history["actions"])
    rewards = discount_rewards(history["rewards"])
    buffer_size = len(observations)
    batches = round(buffer_size/model.batch_size)

    sess.run(model.iterator.initializer, feed_dict={model.buffer_size: buffer_size,
                                                    model.observations_sym: observations,
                                                    model.actions_sym: actions,
                                                    model.rewards_sym: rewards})
    sess.run(model.zero_ops)
    for i in range(batches):
        observation, actions, rewards = sess.run(model.next_batch)
        sess.run(model.accum_ops, feed_dict={
            model.observations_sym: observation,
            model.actions_sym: actions,
            model.rewards_sym: rewards
        })
    sess.run(model.train_step)
