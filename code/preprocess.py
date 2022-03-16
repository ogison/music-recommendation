import dateutil.parser
import argparse
import pickle
import numpy as np
import os
import time

runtime = time.time()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='lastfm', help='path of the dataset')
parser.add_argument('--SESSION_TIMEDELTA', dest='SESSION_TIMEDELTA', default=60*30, help='Session splitting threshold')
parser.add_argument('--MAX_SESSION_LENGTH', dest='MAX_SESSION_LENGTH', default=20, help='Maximum session')
parser.add_argument('--MAX_SESSION_LENGTH_PRE_SPLIT', dest='MAX_SESSION_LENGTH_PRE_SPLIT', default=2, help='Threshold for truncated sessions')
parser.add_argument('--MINIMUM_REQUIRED_SESSIONS', dest='MINIMUM_REQUIRED_SESSIONS', default=3, help='Minimum session')

args = parser.parse_args()

# current directory
home = os.getcwd()

# Here you can change the path to the dataset
DATASET_DIR = home + '/datasets/'+args.dataset_dir
DATASET_FILE = DATASET_DIR + '/userid-timestamp-artid-artname-traid-traname.tsv'
DATASET_W_CONVERTED_TIMESTAMPS = DATASET_DIR + '/1_converted_timestamps.pickle'
DATASET_USER_ARTIST_MAPPED = DATASET_DIR + '/2_user_artist_mapped.pickle'
DATASET_USER_SESSIONS = DATASET_DIR + '/3_user_sessions.pickle'
DATASET_TRAIN_TEST_SPLIT = DATASET_DIR + '/4_train_test_split.pickle'


def file_exists(filename):
    return os.path.isfile(filename)

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))

#'/userid-timestamp-artid-artname-traid-traname.tsv'
def convert_timestamps_lastfm():
    dataset_list = []
    with open(DATASET_FILE, 'rt', buffering=10000, encoding='utf8') as dataset:
        for line in dataset:
            line = line.split('\t')
            user_id     = line[0]
            timestamp   = (dateutil.parser.parse(line[1])).timestamp()
            artist_id   = line[2]
            dataset_list.append( [user_id, timestamp, artist_id] )
            print(user_id)

    #リストを逆順に
    dataset_list = list(reversed(dataset_list))

    save_pickle(dataset_list, DATASET_W_CONVERTED_TIMESTAMPS)

#IDづけ
def map_user_and_artist_id_to_labels():
    dataset_list = load_pickle(DATASET_W_CONVERTED_TIMESTAMPS)

    artist_map = {}
    user_map = {}
    artist_is = ''
    user_id = ''
    for i in range(len(dataset_list)):
        user_id = dataset_list[i][0]
        artist_id = dataset_list[i][2]

        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if artist_id not in artist_map:
            artist_map[artist_id] = len(artist_map)

        dataset_list[i][0] = user_map[user_id]
        dataset_list[i][2] = artist_map[artist_id]
    # Save to pickle file
    save_pickle(dataset_list, DATASET_USER_ARTIST_MAPPED)

def split_single_session(session):
    #42→20,20,2
    splitted = [session[i:i+args.MAX_SESSION_LENGTH] for i in range(0, len(session), args.MAX_SESSION_LENGTH)]
    #最後の分割したセッションが2個以下の場合
    if len(splitted[-1]) < 2:
        #削除
        del splitted[-1]

    return splitted

def perform_session_splits(sessions):
    splitted_sessions = []
    for session in sessions:
        splitted_sessions += split_single_session(session)

    return splitted_sessions

def split_long_sessions(user_sessions):
    for k, v in user_sessions.items():
        user_sessions[k] = perform_session_splits(v)

#セッションひとつずつ
def collapse_session(session):
    new_session = [session[0]]
    for i in range(1, len(session)):
        last_event = new_session[-1]
        current_event = session[i]
        if current_event[1] != last_event[1]:
            new_session.append(current_event)

    return new_session

def collapse_repeating_items(user_sessions):
    for k, sessions in user_sessions.items():
        for i in range(len(sessions)):
            sessions[i] = collapse_session(sessions[i])


''' Splits sessions according to inactivity (time between two consecutive
    actions) and assign sessions to their user. Sessions should be sorted,
    both eventwise internally and compared to other sessions, but this should
    be automatically handled since the dataset is presorted
'''
def sort_and_split_usersessions():
    MAX_SESSION_LENGTH_PRE_SPLIT = args.MAX_SESSION_LENGTH * args.MAX_SESSION_LENGTH_PRE_SPLIT
    dataset_list = load_pickle(DATASET_USER_ARTIST_MAPPED)
    user_sessions = {}
    music_artist_map = {}
    current_session = []
    for event in dataset_list:
        user_id = event[0]
        timestamp = int(event[1])
        music = event[2]

        new_event = [timestamp, music]

        # if new user -> new session
        if user_id not in user_sessions:
            user_sessions[user_id] = []
            current_session = []
            user_sessions[user_id].append(current_session)
            current_session.append(new_event)
            continue

        # it is an existing user: is it a new session?
        # we also know that the current session contains at least one event
        # NB: Dataset is presorted from newest to oldest events
        last_event = current_session[-1]
        last_timestamp = last_event[0]
        timedelta = timestamp - last_timestamp

        if timedelta < args.SESSION_TIMEDELTA:
            # new event belongs to current session
            current_session.append(new_event)
        else:
            # new event belongs to new session
            current_session = [new_event]
            user_sessions[user_id].append(current_session)

    # collapse_repeating_items(user_sessions)

    # Remove sessions that only contain one event
    # Bad to remove stuff from the lists we are iterating through, so create
    # a new datastructure and copy over what we want to keep
    new_user_sessions = {}
    for k in user_sessions.keys():
        if k not in new_user_sessions:
            new_user_sessions[k] = []

        us = user_sessions[k]
        #少なすぎる、多すぎるセッションは除外
        for session in us:
            if len(session) > 1 and len(session) < MAX_SESSION_LENGTH_PRE_SPLIT:
                new_user_sessions[k].append(session)

    # Split too long sessions, before removing user with too few sessions
    #  because splitting can result in more sessions.

    split_long_sessions(new_user_sessions)

    # Remove users with less than 3 session
    # Find users with less than 3 sessions first
    to_be_removed = []
    for k, v in new_user_sessions.items():
        if len(v) < args.MINIMUM_REQUIRED_SESSIONS:
            to_be_removed.append(k)
    # Remove the users we found
    for user in to_be_removed:
        new_user_sessions.pop(user)


    # Do a remapping to account for removed data
    print("remapping to account for removed data...")

    # remap users
    nus = {}
    for k, v in new_user_sessions.items():
        nus[len(nus)] = new_user_sessions[k]

    # remap musicIDs
    mus = {}
    for k, v in nus.items():
        for session in v:
            for i in range(len(session)):
                a = session[i][1]
                if a not in mus:
                    mus[a] = len(mus)+1
                session[i][1] = mus[a]

    save_pickle(nus, DATASET_USER_SESSIONS)


def get_session_lengths(dataset):
    session_lengths = {}
    for k, v in dataset.items():
        session_lengths[k] = []
        for session in v:
            session_lengths[k].append(len(session)-1)

    return session_lengths

def create_padded_sequence(session):
    if len(session) == args.MAX_SESSION_LENGTH:
        return session

    dummy_timestamp = 0
    dummy_label = 0
    length_to_pad = args.MAX_SESSION_LENGTH - len(session)
    padding = [[dummy_timestamp, dummy_label, dummy_label]] * length_to_pad
    session += padding
    return session

def pad_sequences(dataset):
    for k, v in dataset.items():
        for session_index in range(len(v)):
            dataset[k][session_index] = create_padded_sequence(dataset[k][session_index])

# Splits the dataset into a training and a testing set, by extracting the last
# sessions of each user into the test set
def split_to_training_and_testing():
    dataset = load_pickle(DATASET_USER_SESSIONS)
    trainset = {}
    testset = {}

    for k, v in dataset.items():
        n_sessions = len(v)
        # split_point = int(0.8*n_sessions)
        split_point = int(0.8*n_sessions)


        # runtime check to ensure that we have enough sessions for training and testing
        if split_point < 2:
            raise ValueError('User '+str(k)+' with '+str(n_sessions)+""" sessions,
                resulted in split_point: '+str(split_point)+' which gives too
                few training sessions. Please check that data and preprocessing
                is correct.""")

        trainset[k] = v[:split_point]
        testset[k] = v[split_point:]

    # Also need to know session lengths for train- and testset
    train_session_lengths = get_session_lengths(trainset)
    test_session_lengths = get_session_lengths(testset)

    # Finally, pad all sequences before storing everything
    pad_sequences(trainset)
    pad_sequences(testset)

    # Put everything we want to store in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset
    pickle_dict['testset'] = testset
    pickle_dict['train_session_lengths'] = train_session_lengths
    pickle_dict['test_session_lengths'] = test_session_lengths

    save_pickle(pickle_dict , DATASET_TRAIN_TEST_SPLIT)

if __name__ == '__main__':
    if not file_exists(DATASET_W_CONVERTED_TIMESTAMPS):
        print("Converting timestamps.")
        convert_timestamps_lastfm()

    if not file_exists(DATASET_USER_ARTIST_MAPPED):
        print("Mapping user and artist IDs to labels.")
        map_user_and_artist_id_to_labels()

    if not file_exists(DATASET_USER_SESSIONS):
        print("Sorting sessions to users.")
        sort_and_split_usersessions()

    if not file_exists(DATASET_TRAIN_TEST_SPLIT):
        print("Splitting dataset into training and testing sets.")
        split_to_training_and_testing()

    print("Runtime:", str(time.time()-runtime))
