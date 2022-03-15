import tensorflow as tf
import datetime
import argparse
import os
import time
import math
import numpy as np
from utils import PlainRNNDataHandler
from test_util import Tester

class GRU(object):

    def __init__(self, args , datahandler):

        self.learning_rate = args.learning_rate
        self.dropout_late = args.dropout_late
        self.recurrent_dropout_late = args.recurrent_dropout_late
        self.GRU_hidden = args.GRU_hidden
        self.EMBEDDING_SIZE = args.EMBEDDING_SIZE
        self.save_best = args.save_best
        self.MAX_EPOCHS = args.MAX_EPOCHS
        self.BATCHSIZE = args.BATCHSIZE
        self.dataset_dir = args.dataset_dir
        self.epoch_file = './epoch_file-simple-rnn-'+args.dataset_dir+'.pickle'
        self.epoch = datahandler.get_latest_epoch(self.epoch_file)
        self.datahandler = datahandler
        self.best_recall5 = -1
        self.N_ITEMS = self.datahandler.get_num_items()

        home = os.path.dirname(os.path.abspath(__file__))
        self.save_path = home + '/save/'+self.dataset_dir
        self.save_final_dir = self.save_path + '/final/'
        self.save_best_dir = self.save_path + '/best/'
        self.checkpoint_file = "cp-{}.ckpt".format(self.epoch)

        if not os.path.exists("save"):
            os.mkdir("save")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.save_best_dir):
            os.mkdir(self.save_best_dir)
        if not os.path.exists(self.save_final_dir):
            os.mkdir(self.save_final_dir)

        N_ITEMS = self.datahandler.get_num_items()
        self.num_training_batches = self.datahandler.get_num_training_batches()
        self.num_test_batches = self.datahandler.get_num_test_batches()

        self._build_model(args)


    def _build_model(self,args):
        
        self.model = tf.keras.models.Sequential([
          tf.keras.layers.Embedding(self.N_ITEMS, args.EMBEDDING_SIZE,mask_zero=True,trainable=True),
          tf.keras.layers.GRU(args.GRU_hidden,return_sequences=True,activation=None,dropout=args.dropout_late,recurrent_dropout=args.recurrent_dropout_late),
          tf.keras.layers.Dense(self.N_ITEMS),
        ], name= "my_model")

        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,epsilon=1e-08)

        if self.epoch > 0:
            print("|--Restoring model.")
            latest = tf.train.latest_checkpoint(save_best_dir)
            self.model.load_weights(latest)
        self.epoch += 1

    def custom_loss(self,y_val, y_pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_val)    # [ BATCHSIZE x SEQLEN ]
        mask = tf.math.sign(tf.compat.v1.to_float(y_val))
        masked_loss = mask * loss
        return masked_loss

    def train(self,args):
        print("Starting training.")
        print("|-Starting on epoch", self.epoch)
        while self.epoch <= args.MAX_EPOCHS:
            print("Starting epoch #"+str(self.epoch))
            self.epoch_loss = 0

            if args.do_training:
                self.datahandler.reset_user_batch_data()
                _batch_number = 0
                xinput, targetvalues, sl = self.datahandler.get_next_train_batch()

                while len(xinput) > int(self.BATCHSIZE/2):
                    _batch_number += 1
                    batch_start_time = time.time()
                    x_size = len(xinput)

                    # X_embed = tf.nn.embedding_lookup(W_embed, xinput)
                    Y_flat_target = tf.reshape(targetvalues, [-1])

                    with tf.GradientTape() as tape:
                        logits = self.model(tf.convert_to_tensor(xinput), training=True)
                        logits = tf.reshape(logits,[-1,self.N_ITEMS])
                        loss_value = self.custom_loss(Y_flat_target, logits)
                        loss_value = tf.reshape(loss_value, [x_size, -1])            # [ BATCHSIZE, SEQLEN ]

                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


                    loss_value = tf.reshape(loss_value, [x_size, -1])
                    # Stats
                    # Average sequence loss
                    seqloss = tf.reduce_mean(loss_value, 1)
                    batchloss = tf.reduce_mean(seqloss)
                    batch_runtime = time.time() - batch_start_time
                    self.epoch_loss += batchloss

                    if _batch_number%100==0:
                        print("Batch number:", str(_batch_number), "/", str(self.num_training_batches), "| Batch time:", "%.2f" % batch_runtime, " seconds", end='')
                        print(" | Batch loss:", batchloss, end='')
                        eta = (batch_runtime*(self.num_training_batches-_batch_number))/60
                        eta = "%.2f" % eta
                        print(" | ETA:", eta, "minutes.")

                    xinput, targetvalues, sl = self.datahandler.get_next_train_batch()

                print("Epoch", self.epoch, "finished")
                print("|- Epoch loss:", self.epoch_loss)

            self.model.save_weights(os.path.join(self.save_best_dir,self.checkpoint_file))

            self.test(args)

    def test(self,args):
        ##
        ##  TESTING
        ##
        print("Starting testing")
        tester = Tester()
        self.datahandler.reset_user_batch_data()
        _batch_number = 0
        xinput, targetvalues, sl = self.datahandler.get_next_test_batch()
        while len(xinput) > int(self.BATCHSIZE/2):
            batch_start_time = time.time()
            _batch_number += 1
            x_size = len(xinput)

            Y_flat_target = tf.reshape(targetvalues, [-1])
            logits = self.model.predict(tf.convert_to_tensor(xinput))

            top_k_values, top_k_predictions = tf.nn.top_k(logits, k=args.TOP_K)        # [BATCHSIZE x SEQLEN, TOP_K]
            Y_prediction = tf.reshape(top_k_predictions, [x_size, -1, args.TOP_K], name='YTopKPred') #[batchsize, SEQLEN, TOP_K]

            # Evaluate predictions
            tester.evaluate_batch(Y_prediction, targetvalues, sl)

            # Print some stats during testing
            batch_runtime = time.time() - batch_start_time
            if _batch_number%100==0:
                print("Batch number:", str(_batch_number), "/", str(self.num_test_batches), "| Batch time:", "%.2f" % batch_runtime, " seconds", end='')
                eta = (batch_runtime*(self.num_test_batches-_batch_number))/60
                eta = "%.2f" % eta
                print(" ETA:", eta, "minutes.")

            xinput, targetvalues, sl = self.datahandler.get_next_test_batch()

        # Print final test stats for epoch
        test_stats, current_recall5, current_recall10, current_recall20, current_mrr5, current_mrr10, current_mrr20 = tester.get_stats_and_reset()
        print("Recall@5 = " + str(current_recall5))
        print("Recall@10 = " + str(current_recall10))
        print("Recall@20 = " + str(current_recall20))
        print("MRR@5 = " + str(current_mrr5))
        print("MRR@10 = " + str(current_mrr10))
        print("MRR@20 = " + str(current_mrr20))
        self.model.save_weights(os.path.join(self.save_final_dir,self.checkpoint_file))

        if self.save_best:
            if current_recall5 > self.best_recall5:
                # Save the model
                print("Saving model.")
                model.save_weights(os.path.join(self.save_best_dir,self.checkpoint_file))
                print("|- Model saved in file:",self.save_best_dir)

                self.best_recall5 = current_recall5

                self.datahandler.store_current_epoch(self.epoch, self.epoch_file)
                self.datahandler.log_test_stats(self.epoch, self.epoch_loss, test_stats)

        self.epoch += 1
