import tensorflow as tf
import datetime
import argparse
import os
import time
import math
import numpy as np
from utils import PlainRNNDataHandler
from test_util import Tester
from model import GRU

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='lastfm', help='path of the dataset')
parser.add_argument('--save_best', dest='save_best', default=True, help='Whether to save the model parameters')
parser.add_argument('--do_training', dest='do_training', default=True, help='Training or Test')
parser.add_argument('--BATCHSIZE', dest='BATCHSIZE', default=100, help='Batch size')
parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, help='Learning_rate of Adam')
parser.add_argument('--dropout_late', dest='dropout_late', default=0.1, help='dropout rate')
parser.add_argument('--recurrent_dropout_late', dest='recurrent_dropout_late', default=0.15, help='Recurrent dropout rate')
parser.add_argument('--SEQLEN', dest='SEQLEN', default=20-1, help='maximum number of actions in a session')
parser.add_argument('--GRU_hidden', dest='GRU_hidden', default= 100, help='GRU hidden layers')
parser.add_argument('--EMBEDDING_SIZE', dest='EMBEDDING_SIZE', default= 100, help='EMBEDDING dimension')
parser.add_argument('--TOP_K', dest='TOP_K', default=20, help='Evaluation TopK')
parser.add_argument('--MAX_EPOCHS', dest='MAX_EPOCHS', default=20, help='Maximum epoch')

args = parser.parse_args()

if __name__ == '__main__':
    # Specify path to dataset here
    home = os.getcwd()
    dataset_path = home + '/datasets/'+args.dataset_dir+'/4_train_test_split.pickle'
    date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
    log_file = './testlog/'+str(date_now)+'-testing-plain-rnn.txt'

    seed = 0
    tf.random.set_seed(seed)

    # Load training data
    datahandler = PlainRNNDataHandler(dataset_path, args.BATCHSIZE, log_file)
    # Number of item types
    N_ITEMS = -1
    N_ITEMS = datahandler.get_num_items()
    # Total number of sessions
    N_SESSIONS = datahandler.get_num_training_sessions()

    message = "------------------------------------------------------------------------\n"
    message += "DATASET: "+args.dataset_dir+" MODEL: plain RNN"
    message += "\nCONFIG: N_ITEMS="+str(N_ITEMS)+" BATCHSIZE="+str(args.BATCHSIZE)+" GRU_hidden="+str(args.GRU_hidden)
    message += "\nSEQLEN="+str(args.SEQLEN)+" EMBEDDING_SIZE="+str(args.EMBEDDING_SIZE)
    message += "\nN_SESSIONS="+str(N_SESSIONS)+" SEED="+str(seed)+"\n"
    datahandler.log_config(message)
    print(message)

    ##
    ## The model
    ##
    print("Creating model")

    model = GRU(args,datahandler)

    model.train(args) if args.do_training == True else model.test(args)
