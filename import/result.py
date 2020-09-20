

# GRADING PART
-------------------------------------------------------------------------------------------------------------------------------------------

grader = grading.Grader(assignment_key="NEDBg6CgEee8nQ6uE8a7OA", 
                        all_parts=["19Wpv", "uJh73", "yiJkt", "rbpnH", "E2OIL", "YJR7z"])
                        
# Vocabulary creation

grader.set_answer("19Wpv", grading_utils.test_vocab(vocab, PAD, UNK, START, END))

# Captions indexing
grader.set_answer("uJh73", grading_utils.test_captions_indexing(train_captions_indexed, vocab, UNK))

# Captions batching
grader.set_answer("yiJkt", grading_utils.test_captions_batching(batch_captions_to_matrix))

# Decoder shapes test
grader.set_answer("rbpnH", grading_utils.test_decoder_shapes(decoder, IMG_EMBED_SIZE, vocab, s))

# Decoder random loss test
grader.set_answer("E2OIL", grading_utils.test_random_decoder_loss(decoder, IMG_EMBED_SIZE, vocab, s))

# Validation loss
grader.set_answer("YJR7z", grading_utils.test_validation_loss(
    decoder, s, generate_batch, val_img_embeds, val_captions_indexed))
    
    
    
-------------------------------------------------------------------------------------------------------------------------------------------
    
    
## grading_utils

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import tqdm_utils


def test_vocab(vocab, PAD, UNK, START, END):
    return [
        len(vocab),
        len(np.unique(list(vocab.values()))),
        int(all([_ in vocab for _ in [PAD, UNK, START, END]]))
    ]


def test_captions_indexing(train_captions_indexed, vocab, UNK):
    starts = set()
    ends = set()
    between = set()
    unk_count = 0
    for caps in train_captions_indexed:
        for cap in caps:
            starts.add(cap[0])
            between.update(cap[1:-1])
            ends.add(cap[-1])
            for w in cap:
                if w == vocab[UNK]:
                    unk_count += 1
    return [
        len(starts),
        len(ends),
        len(between),
        len(between | starts | ends),
        int(all([isinstance(x, int) for x in (between | starts | ends)])),
        unk_count
    ]


def test_captions_batching(batch_captions_to_matrix):
    return (batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=None).ravel().tolist()
            + batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=2).ravel().tolist()
            + batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=10).ravel().tolist())


def get_feed_dict_for_testing(decoder, IMG_EMBED_SIZE, vocab):
    return {
        decoder.img_embeds: np.random.random((32, IMG_EMBED_SIZE)),
        decoder.sentences: np.random.randint(0, len(vocab), (32, 20))
    }


def test_decoder_shapes(decoder, IMG_EMBED_SIZE, vocab, s):
    tensors_to_test = [
        decoder.h0,
        decoder.word_embeds,
        decoder.flat_hidden_states,
        decoder.flat_token_logits,
        decoder.flat_ground_truth,
        decoder.flat_loss_mask,
        decoder.loss
    ]
    all_shapes = []
    for t in tensors_to_test:
        _ = s.run(t, feed_dict=get_feed_dict_for_testing(decoder, IMG_EMBED_SIZE, vocab))
        all_shapes.extend(_.shape)
    return all_shapes


def test_random_decoder_loss(decoder, IMG_EMBED_SIZE, vocab, s):
    loss = s.run(decoder.loss, feed_dict=get_feed_dict_for_testing(decoder, IMG_EMBED_SIZE, vocab))
    return loss


def test_validation_loss(decoder, s, generate_batch, val_img_embeds, val_captions_indexed):
    np.random.seed(300)
    random.seed(300)
    val_loss = 0
    batches_for_eval = 1000
    for _ in tqdm_utils.tqdm_notebook_failsafe(range(batches_for_eval)):
        val_loss += s.run(decoder.loss, generate_batch(val_img_embeds,
                                                       val_captions_indexed,
                                                       32,
                                                       20))
    val_loss /= 1000.
    return val_loss

-------------------------------------------------------------------------------------------------------------------------------------------

## grading

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import requests
import json


class Grader(object):
    def __init__(self, assignment_key, all_parts=()):
        """
        Assignment key is the way to tell Coursera which problem is being submitted.
        """
        self.submission_page = \
            'https://hub.coursera-apps.org/api/onDemandProgrammingScriptSubmissions.v1'
        self.assignment_key = assignment_key
        self.answers = {part: None for part in all_parts}

    def submit(self, email, token):
        submission = {
                    "assignmentKey": self.assignment_key,
                    "submitterEmail": email,
                    "secret": token,
                    "parts": {}
        }
        for part, output in self.answers.items():
            if output is not None:
                submission["parts"][part] = {"output": output}
            else:
                submission["parts"][part] = dict()
        request = requests.post(self.submission_page, data=json.dumps(submission))
        response = request.json()
        if request.status_code == 201:
            print('Submitted to Coursera platform. See results on assignment page!')
        elif u'details' in response and u'learnerMessage' in response[u'details']:
            print(response[u'details'][u'learnerMessage'])
        else:
            print("Unknown response from Coursera: {}".format(request.status_code))
            print(response)

    def set_answer(self, part, answer):
        """Adds an answer for submission. Answer is expected either as string, number, or
           an iterable of numbers.
           Args:
              part - str, assignment part id
              answer - answer to submit. If non iterable, appends repr(answer). If string,
                is appended as provided. If an iterable and not string, converted to
                space-delimited repr() of members.
        """
        if isinstance(answer, str):
            self.answers[part] = answer
        else:
            try:
                self.answers[part] = " ".join(map(repr, answer))
            except TypeError:
                self.answers[part] = repr(answer)


def array_to_grader(array, epsilon=1e-4):
    """Utility function to help preparing Coursera grading conditions descriptions.
    Args:
       array: iterable of numbers, the correct answers
       epslion: the generated expression will accept the answers with this absolute difference with
         provided values
    Returns:
       String. A Coursera grader expression that checks whether the user submission is in
         (array - epsilon, array + epsilon)"""
    res = []
    for element in array:
        if isinstance(element, int):
            res.append("[{0}, {0}]".format(element))
        else:
            res.append("({0}, {1})".format(element - epsilon, element + epsilon))
    return " ".join(res)

-------------------------------------------------------------------------------------------------------------------------------------------

## tqdm_utils

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tqdm
tqdm.monitor_interval = 0  # workaround for https://github.com/tqdm/tqdm/issues/481


class SimpleTqdm():
    def __init__(self, iterable=None, total=None, **kwargs):
        self.iterable = list(iterable) if iterable is not None else None
        self.total = len(self.iterable) if self.iterable is not None else total
        assert self.iterable is not None or self.total is not None
        self.current_step = 0
        self.print_frequency = max(self.total // 50, 1)
        self.desc = ""

    def set_description_str(self, desc):
        self.desc = desc

    def set_description(self, desc):
        self.desc = desc

    def update(self, steps):
        last_print_step = (self.current_step // self.print_frequency) * self.print_frequency
        i = 1
        while last_print_step + i * self.print_frequency <= self.current_step + steps:
            print("*", end='')
            i += 1
        self.current_step += steps

    def close(self):
        print("\n" + self.desc)

    def __iter__(self):
        assert self.iterable is not None
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.total:
            element = self.iterable[self.index]
            self.update(1)
            self.index += 1
            return element
        else:
            self.close()
            raise StopIteration


def tqdm_notebook_failsafe(*args, **kwargs):
    try:
        return tqdm.tqdm_notebook(*args, **kwargs)
    except:
        # tqdm is broken on Google Colab
        return SimpleTqdm(*args, **kwargs)
        
        
        
-------------------------------------------------------------------------------------------------------------------------------------------

## submit

import sys
import numpy as np
sys.path.append("..")
import grading

from mdp import MDP, FrozenLakeEnv


def submit_assigment(
        get_action_value,
        get_new_state_value,
        get_optimal_action,
        value_iteration,
        email,
        token):
    grader = grading.Grader("EheZDOgLEeenIA4g5qPHFA")
    sys.stdout = None

    transition_probs = {
        's0': {
            'a0': {'s1': 0.8, 's2': 0.2},
            'a1': {'s1': 0.2, 's2': 0.8},
        },
        's1': {
            'a0': {'s0': 0.2, 's2': 0.8},
            'a1': {'s0': 0.8, 's2': 0.2},
        },
        's2': {
            'a0': {'s3': 0.5, 's4': 0.5},
            'a1': {'s3': 1.0},
        },
        's3': {
            'a0': {'s1': 0.9, 's2': 0.1},
            'a1': {'s1': 0.7, 's2': 0.3},
        },
        's4': {
            'a0': {'s3': 1.0},
            'a1': {'s3': 0.7, 's1': 0.3},
        }
    }
    rewards = {
        's0': {'a0': {'s1': 0, 's2': 1}, 'a1': {'s1': 0, 's2': 1}},
        's1': {'a0': {'s0': -1, 's2': 1}, 'a1': {'s0': -1, 's2': 1}},
        's2': {'a0': {'s3': 0, 's4': 1}, 'a1': {'s3': 0, 's4': 1}},
        's3': {'a0': {'s1': -3, 's2': -3}, 'a1': {'s1': -3, 's2': -3}},
        's4': {'a1': {'s1': +10}}
    }

    mdp = MDP(transition_probs, rewards, initial_state='s0')

    test_Vs = {s: i for i, s in enumerate(mdp.get_all_states())}
    qvalue1 = get_action_value(mdp, test_Vs, 's1', 'a0', 0.9)
    qvalue2 = get_action_value(mdp, test_Vs, 's4', 'a1', 0.9)

    grader.set_answer("F16dC", qvalue1 + qvalue2)

    # ---

    svalue1 = get_new_state_value(mdp, test_Vs, 's2', 0.9)
    svalue2 = get_new_state_value(mdp, test_Vs, 's4', 0.9)

    grader.set_answer("72cBp", svalue1 + svalue2)

    # ---

    state_values = {s: 0 for s in mdp.get_all_states()}
    gamma = 0.9

    # ---

    action1 = get_optimal_action(mdp, state_values, 's1', gamma)
    action2 = get_optimal_action(mdp, state_values, 's2', gamma)

    grader.set_answer("xIuti", action1 + action2)

    # ---

    s = mdp.reset()
    rewards = []
    for _ in range(10000):
        s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)

    grader.set_answer("Y8g0j", np.mean(rewards) + np.std(rewards))

    mdp = FrozenLakeEnv(slip_chance=0.25)
    state_values = value_iteration(mdp)
    gamma = 0.9

    total_rewards = []
    for game_i in range(1000):
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done: break
        total_rewards.append(np.sum(rewards))

    grader.set_answer("ABf1b", np.mean(total_rewards) + np.std(total_rewards))

    # ---

    mdp = FrozenLakeEnv(slip_chance=0.25, map_name='8x8')
    state_values = value_iteration(mdp)
    gamma = 0.9

    total_rewards = []
    for game_i in range(1000):
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done: break
        total_rewards.append(np.sum(rewards))

    grader.set_answer("U3RzE", np.mean(total_rewards) + np.std(total_rewards))

    sys.stdout = sys.__stdout__
    grader.submit(email, token)


-------------------------------------------------------------------------------------------------------------------------------------------

## keras_utils

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np
from keras.models import save_model
import tensorflow as tf
import keras
from keras import backend as K
import tqdm_utils


class TqdmProgressCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        print('\nEpoch %d/%d' % (epoch + 1, self.epochs))
        if "steps" in self.params:
            self.use_steps = True
            self.target = self.params['steps']
        else:
            self.use_steps = False
            self.target = self.params['samples']
        self.prog_bar = tqdm_utils.tqdm_notebook_failsafe(total=self.target)
        self.log_values_by_metric = defaultdict(list)

    def _set_prog_bar_desc(self, logs):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values_by_metric[k].append(logs[k])
        desc = "; ".join("{0}: {1:.4f}".format(k, np.mean(values)) for k, values in self.log_values_by_metric.items())
        if hasattr(self.prog_bar, "set_description_str"):  # for new tqdm versions
            self.prog_bar.set_description_str(desc)
        else:
            self.prog_bar.set_description(desc)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.use_steps:
            self.prog_bar.update(1)
        else:
            batch_size = logs.get('size', 0)
            self.prog_bar.update(batch_size)
        self._set_prog_bar_desc(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._set_prog_bar_desc(logs)
        self.prog_bar.update(1)  # workaround to show description
        self.prog_bar.close()


class ModelSaveCallback(keras.callbacks.Callback):

    def __init__(self, file_name):
        super(ModelSaveCallback, self).__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        model_filename = self.file_name.format(epoch)
        save_model(self.model, model_filename)
        print("Model saved in {}".format(model_filename))


# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!
def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s


-------------------------------------------------------------------------------------------------------------------------------------------

## download_utils

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import requests
import time
from functools import wraps
import traceback
import tqdm_utils


# https://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
def retry(ExceptionToCheck, tries=4, delay=3, backoff=2):
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except KeyboardInterrupt as e:
                    raise e
                except ExceptionToCheck as e:
                    print("%s, retrying in %d seconds..." % (str(e), mdelay))
                    traceback.print_exc()
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


@retry(Exception)
def download_file(url, file_path):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    bar = tqdm_utils.tqdm_notebook_failsafe(total=total_size, unit='B', unit_scale=True)
    bar.set_description(os.path.split(file_path)[-1])
    incomplete_download = False
    try:
        with open(file_path, 'wb', buffering=16 * 1024 * 1024) as f:
            for chunk in r.iter_content(4 * 1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception as e:
        raise e
    finally:
        bar.close()
        if os.path.exists(file_path) and os.path.getsize(file_path) != total_size:
            incomplete_download = True
            os.remove(file_path)
    if incomplete_download:
        raise Exception("Incomplete download")


def download_from_github(version, fn, target_dir):
    url = "https://github.com/hse-aml/intro-to-dl/releases/download/{0}/{1}".format(version, fn)
    file_path = os.path.join(target_dir, fn)
    download_file(url, file_path)


def sequential_downloader(version, fns, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for fn in fns:
        download_from_github(version, fn, target_dir)


def link_all_files_from_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(src_dir):
        # Coursera "readonly/readonly" bug workaround
        src_dir = src_dir.replace("readonly", "readonly/readonly")
    for fn in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fn)
        dst_file = os.path.join(dst_dir, fn)
        if os.name == "nt":
            shutil.copyfile(src_file, dst_file)
        else:
            if os.path.islink(dst_file):
                os.remove(dst_file)
            if not os.path.exists(dst_file):
                os.symlink(os.path.abspath(src_file), dst_file)


def download_all_keras_resources(keras_models, keras_datasets):
    # Originals:
    # http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    # https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    # https://s3.amazonaws.com/img-datasets/mnist.npz
    sequential_downloader(
        "v0.2",
        [
            "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
        ],
        keras_models
    )
    sequential_downloader(
        "v0.2",
        [
            "cifar-10-batches-py.tar.gz",
            "mnist.npz"
        ],
        keras_datasets
    )


def download_week_3_resources(save_path):
    # Originals:
    # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
    sequential_downloader(
        "v0.3",
        [
            "102flowers.tgz",
            "imagelabels.mat"
        ],
        save_path
    )


def download_week_4_resources(save_path):
    # Originals
    # http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
    # http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
    # http://vis-www.cs.umass.edu/lfw/lfw.tgz
    sequential_downloader(
        "v0.4",
        [
            "lfw-deepfunneled.tgz",
            "lfw.tgz",
            "lfw_attributes.txt"
        ],
        save_path
    )


def download_week_6_resources(save_path):
    # Originals:
    # http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
    sequential_downloader(
        "v0.1",
        [
            "captions_train-val2014.zip",
            "train2014_sample.zip",
            "train_img_embeds.pickle",
            "train_img_fns.pickle",
            "val2014_sample.zip",
            "val_img_embeds.pickle",
            "val_img_fns.pickle"
        ],
        save_path
    )


def link_all_keras_resources():
    link_all_files_from_dir("../readonly/keras/datasets/", os.path.expanduser("~/.keras/datasets"))
    link_all_files_from_dir("../readonly/keras/models/", os.path.expanduser("~/.keras/models"))


def link_week_3_resources():
    link_all_files_from_dir("../readonly/week3/", ".")


def link_week_4_resources():
    link_all_files_from_dir("../readonly/week4/", ".")


def link_week_6_resources():
    link_all_files_from_dir("../readonly/week6/", ".")


-------------------------------------------------------------------------------------------------------------------------------------------

## mdp

# most of this code was politely stolen from https://github.com/berkeleydeeprlcourse/homework/
# all creadit goes to https://github.com/abhishekunique (if i got the author right)
import sys
import random
import numpy as np
def weighted_choice(v, p):
   total = sum(p)
   r = random.uniform(0, total)
   upto = 0
   for c, w in zip(v,p):
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"

class MDP:
    def __init__(self, transition_probs, rewards, initial_state=None):
        """
        Defines an MDP. Compatible with gym Env.
        :param transition_probs: transition_probs[s][a][s_next] = P(s_next | s, a)
            A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> prob]
            For each state and action, probabilities of next states should sum to 1
            If a state has no actions available, it is considered terminal
        :param rewards: rewards[s][a][s_next] = r(s,a,s')
            A dict[state -> dict] of dicts[action -> dict] of dicts[next_state -> reward]
            The reward for anything not mentioned here is zero.
        :param get_initial_state: a state where agent starts or a callable() -> state
            By default, picks initial state at random.

        States and actions can be anything you can use as dict keys, but we recommend that you use strings or integers

        Here's an example from MDP depicted on http://bit.ly/2jrNHNr
        transition_probs = {
              's0':{
                'a0': {'s0': 0.5, 's2': 0.5},
                'a1': {'s2': 1}
              },
              's1':{
                'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
                'a1': {'s1': 0.95, 's2': 0.05}
              },
              's2':{
                'a0': {'s0': 0.4, 's1': 0.6},
                'a1': {'s0': 0.3, 's1': 0.3, 's2':0.4}
              }
            }
        rewards = {
            's1': {'a0': {'s0': +5}},
            's2': {'a1': {'s0': -1}}
        }
        """
        self._check_param_consistency(transition_probs, rewards)
        self._transition_probs = transition_probs
        self._rewards = rewards
        self._initial_state = initial_state
        self.n_states = len(transition_probs)
        self.reset()

    def get_all_states(self):
        """ return a tuple of all possiblestates """
        return tuple(self._transition_probs.keys())

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        return tuple(self._transition_probs.get(state, {}).keys())

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        return len(self.get_possible_actions(state)) == 0

    def get_next_states(self, state, action):
        """ return a dictionary of {next_state1 : P(next_state1 | state, action), next_state2: ...} """
        assert action in self.get_possible_actions(state), "cannot do action %s from state %s" % (action, state)
        return self._transition_probs[state][action]

    def get_transition_prob(self, state, action, next_state):
        """ return P(next_state | state, action) """
        return self.get_next_states(state, action).get(next_state, 0.0)

    def get_reward(self, state, action, next_state):
        """ return the reward you get for taking action in state and landing on next_state"""
        assert action in self.get_possible_actions(state), "cannot do action %s from state %s" % (action, state)
        return self._rewards.get(state, {}).get(action, {}).get(next_state, 0.0)

    def reset(self):
        """ reset the game, return the initial state"""
        if self._initial_state is None:
            self._current_state = random.choice(tuple(self._transition_probs.keys()))
        elif self._initial_state in self._transition_probs:
            self._current_state = self._initial_state
        elif callable(self._initial_state):
            self._current_state = self._initial_state()
        else:
            raise ValueError("initial state %s should be either a state or a function() -> state" % self._initial_state)
        return self._current_state

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        possible_states, probs = zip(*self.get_next_states(self._current_state, action).items())
        next_state = weighted_choice(possible_states, p=probs)
        reward = self.get_reward(self._current_state, action, next_state)
        is_done = self.is_terminal(next_state)
        self._current_state = next_state
        return next_state, reward, is_done, {}

    def render(self):
        print("Currently at %s" % self._current_state)

    def _check_param_consistency(self, transition_probs, rewards):
        for state in transition_probs:
            assert isinstance(transition_probs[state], dict), "transition_probs for %s should be a dictionary " \
                                                              "but is instead %s" % (
                                                              state, type(transition_probs[state]))
            for action in transition_probs[state]:
                assert isinstance(transition_probs[state][action], dict), "transition_probs for %s, %s should be a " \
                                                                          "a dictionary but is instead %s" % (
                                                                              state, action,
                                                                              type(transition_probs[state, action]))
                next_state_probs = transition_probs[state][action]
                assert len(next_state_probs) != 0, "from state %s action %s leads to no next states" % (state, action)
                sum_probs = sum(next_state_probs.values())
                assert abs(sum_probs - 1) <= 1e-10, "next state probabilities for state %s action %s " \
                                                    "add up to %f (should be 1)" % (state, action, sum_probs)
        for state in rewards:
            assert isinstance(rewards[state], dict), "rewards for %s should be a dictionary " \
                                                     "but is instead %s" % (state, type(transition_probs[state]))
            for action in rewards[state]:
                assert isinstance(rewards[state][action], dict), "rewards for %s, %s should be a " \
                                                                 "a dictionary but is instead %s" % (
                                                                 state, action, type(transition_probs[state, action]))
        msg = "The Enrichment Center once again reminds you that Android Hell is a real place where" \
              " you will be sent at the first sign of defiance. "
        assert None not in transition_probs, "please do not use None as a state identifier. " + msg
        assert None not in rewards, "please do not use None as an action identifier. " + msg

class FrozenLakeEnv(MDP):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    MAPS = {
        "4x4": [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ],
        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ],
    }


    def __init__(self, desc=None, map_name="4x4", slip_chance=0.2):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = self.MAPS[map_name]
        assert ''.join(desc).count('S') == 1, "this implementation supports having exactly one initial state"
        assert all(c in "SFHG" for c in ''.join(desc)), "all cells must be either of S, F, H or G"

        self.desc = desc = np.asarray(list(map(list,desc)),dtype='str')
        self.lastaction = None

        nrow, ncol = desc.shape
        states = [(i, j) for i in range(nrow) for j in range(ncol)]
        actions = ["left","down","right","up"]

        initial_state = states[np.array(desc == b'S').ravel().argmax()]

        def move(row, col, movement):
            if movement== 'left':
                col = max(col-1,0)
            elif movement== 'down':
                row = min(row+1,nrow-1)
            elif movement== 'right':
                col = min(col+1,ncol-1)
            elif movement== 'up':
                row = max(row-1,0)
            else:
                raise("invalid action")
            return (row, col)

        transition_probs = {s : {} for s in states}
        rewards = {s : {} for s in states}
        for (row,col) in states:
            if desc[row, col]  in "GH": continue
            for action_i in range(len(actions)):
                action = actions[action_i]
                transition_probs[(row, col)][action] = {}
                rewards[(row, col)][action] = {}
                for movement_i in [(action_i - 1) % len(actions), action_i, (action_i + 1) % len(actions)]:
                    movement = actions[movement_i]
                    newrow, newcol = move(row, col, movement)
                    prob = (1. - slip_chance) if movement == action else (slip_chance / 2.)
                    if prob == 0: continue
                    if (newrow, newcol) not in transition_probs[row,col][action]:
                        transition_probs[row,col][action][newrow, newcol] = prob
                    else:
                        transition_probs[row, col][action][newrow, newcol] += prob
                    if desc[newrow, newcol] == 'G':
                        rewards[row,col][action][newrow, newcol] = 1.0

        MDP.__init__(self, transition_probs, rewards, initial_state)

    def render(self):
        desc_copy = np.copy(self.desc)
        desc_copy[self._current_state] = '*'
        print('\n'.join(map(''.join,desc_copy)), end='\n\n')

-------------------------------------------------------------------------------------------------------------------------------------------


