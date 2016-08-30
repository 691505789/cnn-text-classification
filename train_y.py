#! /usr/bin/env python
# encoding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_loader
from cnn_graph import TextCNN
from tensorflow.contrib import learn
from sklearn import cross_validation
import preprocessing

# 伴随tensorflow的summary和checkout
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 10, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# w2v文件路径
tf.flags.DEFINE_string("w2v_path", "./w2v_model/vectors_50.bin", "w2v file")
tf.flags.DEFINE_string("file_dir","./data_process/jd","train/test dataSet")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
files = ["reviews.neg","reviews.pos"]
# 加载所有的未切分的数据
x_text, y_labels,neg_examples,pos_examples = data_loader.\
    load_data_and_labels(data_dir=FLAGS.file_dir,files=files,splitable=False)

# 获取消极数据的2/3,得到的评论的长度离散度更低
neg_accept_length = preprocessing.freq_factor(neg_examples,
                                         percentage=0.8, drawable=False)
neg_accept_length = [item[0] for item in neg_accept_length]
neg_examples = data_loader.load_data_by_length(neg_examples,neg_accept_length)

# 获取积极数据的2/3,得到的评论的长度离散度更低
pos_accept_length = preprocessing.freq_factor(pos_examples,
                                         percentage=0.8, drawable=False)
pos_accept_length = [item[0] for item in pos_accept_length]
pos_examples = data_loader.load_data_by_length(pos_examples,pos_accept_length)

x_text = neg_examples + pos_examples
neg_labels = [[1,0] for _ in neg_examples]
pos_labels = [[0,1] for _ in pos_examples]
y_labels = np.concatenate([neg_labels,pos_labels], axis=0)
print("Loading data finish")

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text]) # 最长的句子的长度
print(max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# 加载提前训练的w2v数据集
word_vecs = data_loader.load_bin_vec(fname=FLAGS.w2v_path,
                         vocab=list(vocab_processor.vocabulary_._mapping),
                                     ksize=FLAGS.embedding_dim)
# 加载嵌入层的table
W = data_loader.get_W(word_vecs=word_vecs,
                  vocab_ids_map=vocab_processor.vocabulary_._mapping,
                  k=FLAGS.embedding_dim,is_rand=False)

# 随机化数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_labels)))
x_shuffled = x[shuffle_indices]
y_shuffled = y_labels[shuffle_indices]

out_path = os.path.abspath(os.path.join(os.path.curdir, "runs","parameters"))
parameters = "新全连接+jd数据+10\n" \
             "embedding_dim: {},\n" \
             "filter_sizes:{},\n" \
             "num_filters:{},\n" \
             "dropout_keep_prob:{},\n" \
             "l2_reg_lambda:{},\n" \
             "num_epochs:{},\n" \
             "batch_size:{}".format(FLAGS.embedding_dim,FLAGS.filter_sizes,FLAGS.num_filters,
                                    FLAGS.dropout_keep_prob,FLAGS.l2_reg_lambda,FLAGS.num_epochs,
                                    FLAGS.batch_size)
open(out_path, 'w').write(parameters)

# Training
# ==================================================
def train(X_train, X_dev, x_test, y_train, y_dev, y_test):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=max_document_length,
                num_classes=2,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                embedding_table=W,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)


            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                # _, step, loss, accuracy = sess.run(
                #     [train_op, global_step, cnn.loss, cnn.accuracy],
                #     feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                # step, loss, accuracy = sess.run(
                #     [global_step, cnn.loss, cnn.accuracy],
                #     feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)



            # Generate batches
            batches = data_loader.batch_iter(
                list(zip(X_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(X_dev, y_dev, writer=dev_summary_writer)
                    # dev_step(X_dev, y_dev, writer=None)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # Test loop
            # Generate batches for one epoch
            batches = data_loader.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
            # Collect the predictions here
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(cnn.predictions, {cnn.input_x: x_test_batch, cnn.dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            correct_predictions = float(sum(
                all_predictions == np.argmax(y_test,axis=1)))

            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
            # open(os.path.join(out_dir,"test"),'a').write("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
            out_path = os.path.abspath(os.path.join(os.path.curdir, "runs","test"))
            open(out_path,'a').write("{:g},".format(correct_predictions / float(len(y_test))))
            print("\n写入成功！\n")


# cross-validation
kf = cross_validation.KFold(len(x_shuffled), n_folds=3)
for train_index, test_index in kf:
    X_train_total = x_shuffled[train_index]
    y_train_total = y_shuffled[train_index]
    x_test = x_shuffled[test_index]
    y_test = y_shuffled[test_index]

    # 分割训练集与验证集
    X_train, X_dev, y_train, y_dev = cross_validation.train_test_split(
        X_train_total, y_train_total, test_size=0.2, random_state=0)

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    train(X_train,X_dev,x_test,y_train,y_dev,y_test)

# 分割训练集与测试训练集
# X_train_total, X_test, y_train_total, y_test = cross_validation.train_test_split(
#     x_shuffled, y_shuffled, test_size=0.3, random_state=0)
#
# # 分割训练集与验证集
# X_train, X_dev, y_train, y_dev = cross_validation.train_test_split(
#     X_train_total, y_train_total, test_size=0.2, random_state=0)
#
# # x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
# # y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
# train(X_train,X_dev,X_test,y_train,y_dev,y_test)