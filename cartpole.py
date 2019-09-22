import random
import warnings
from collections import deque

import gym
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.80
LEARNING_RATE = 0.001  # unused as we use Experience Replay type of Q-Learning
# See more on Experience Replay here: https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits

MEMORY_SIZE = 5000  # used only for Non-Incremental learning, i.e. partial_fit=False
BATCH_SIZE = 500

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.996

IS_PARTIAL_FIT = True
RENDER = False
SCORE_LOGGER = False
MY_LOGS = True
NO_ITERATIONS = 100

DEFAULT_EPSILON = 0.1

def get_regressor():
    if IS_PARTIAL_FIT:
        return SGDRegressor()

    return AdaBoostRegressor(n_estimators=50)
    # return RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    # return LGBMRegressor(n_estimators=100, n_jobs=-1);

class DQNSolver:

    def __init__(self, action_space, is_partial_fit: bool = False):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self._is_partial_fit = is_partial_fit

        self.predict_mode = False

        regressor = get_regressor()
        self.model = MultiOutputRegressor(regressor)

        self.isFit = False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            # print("Random guess")
            # if self.predict_mode:
            #     self.predict_mode = False;
            #     print("Switched to random mode")
            return random.randrange(self.action_space)

        # print("Predict mode")
        # if not self.predict_mode:
        #     self.predict_mode = True
        #     print("Switched to predict mode")

        if self.isFit == True:
            q_values = self.model.predict(state)
        else:
            q_values = np.zeros(self.action_space).reshape(1, -1)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        if self._is_partial_fit:
            batch = list(self.memory)
            # if we use Incremental Model, we don't have to keep whole memory, as we use just last batch
            # and the rest of the history is stored within the model, indirectly through learning
            self.memory = deque(maxlen=BATCH_SIZE)
        else:
            batch = random.sample(self.memory, int(len(self.memory) / 1))
        X = []
        targets = []
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                if self.isFit:
                    q_update = (reward + GAMMA * np.amax(
                        self.model.predict(state_next)[0]))  # Return the maximum of an array or maximum along an axis
                else:
                    q_update = reward
            if self.isFit:
                q_values = self.model.predict(state)
            else:
                q_values = np.zeros(self.action_space).reshape(1, -1)
            q_values[0][action] = q_update

            X.append(list(state[0]))
            targets.append(q_values[0])

        if self._is_partial_fit:
            self.model.partial_fit(X, targets)
        else:
            self.model.fit(X, targets)

        self.isFit = True
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(action_space, is_partial_fit=IS_PARTIAL_FIT)
    run = 0
    current_max = 0
    while run < NO_ITERATIONS:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        # run = 0
        while True:
            step += 1
            if RENDER:
                env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)

            if MY_LOGS:
                print(state_next, reward, terminal)

            reward = reward if not terminal else -reward

            state_next = np.reshape(state_next,
                                    [1, observation_space])  # Gives a new shape to an array without changing its data.
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            current_max = max(current_max, step)
            if terminal:
                print("Run: " + str(run) + ", exploration_rate: " + str(
                    dqn_solver.exploration_rate) + ", episodes: " + str(step), ", max episodes reached: ",
                      str(current_max))
                if (SCORE_LOGGER):
                    score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cartpole()
