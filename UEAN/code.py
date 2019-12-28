# encoding: utf-8
import tensorflow as tf
import sys
import numpy as np
import time
import os
import logging


# gpu device num
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
train_percent = int(sys.argv[2])
loss_tag = sys.argv[3]


word_least_frequency = 5
logging.info('init dataset ...')
my_dataset = dataset()

user_size = my_dataset.user_dict.__len__()
logging.info('init dataset completed')
################################################## para ##################################################
rnn_hidden_dim = 128
margin = 0.5
batch_pair_size = 10
learning_rate = 0.001
user_dim = rnn_hidden_dim
attention_size = 64

alpha = 1e-8



############################################## place_holder  ##############################################
place_que_words_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="place_que_words_index")
place_que_sen_len = tf.placeholder(dtype=tf.int32, shape=[None], name="place_que_sen_len")


place_positive_ans_words_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="place_positive_ans_words_index")
place_positive_ans_sen_len = tf.placeholder(dtype=tf.int32, shape=[None], name="place_positive_ans_sen_len")
place_positive_ans_len = tf.placeholder(dtype=tf.int32, shape=[None], name="place_positive_ans_len")
place_positive_user_lookup = tf.placeholder(dtype=tf.int32, shape=[None], name="place_positive_user_lookup")
place_positive_ans_indicies = tf.placeholder(tf.int32, shape=[None], name="place_positive_ans_indicies")
place_positive_ans_sen_indicies = tf.placeholder(tf.int32, shape=[None, None], name="place_positive_ans_sen_indicies")
place_positive_ans_sen_padding_indicies = tf.placeholder(tf.int32, shape=[None, None], name="place_positive_ans_sen_padding_indicies")



place_negative_ans_words_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="place_negative_ans_words_index")
place_negative_ans_sen_len = tf.placeholder(dtype=tf.int32, shape=[None], name="place_negative_ans_sen_len")
place_negative_ans_len = tf.placeholder(dtype=tf.int32, shape=[None], name="place_negative_ans_len")
place_negative_user_lookup = tf.placeholder(dtype=tf.int32, shape=[None], name="place_negative_user_lookup")
place_negative_ans_indicies = tf.placeholder(tf.int32, shape=[None], name="place_negative_ans_indicies")
place_negative_ans_sen_indicies = tf.placeholder(tf.int32, shape=[None, None], name="place_negative_ans_sen_indicies")
place_negative_ans_sen_padding_indicies = tf.placeholder(tf.int32, shape=[None, None], name="place_negative_ans_sen_padding_indicies")


############################################## embedding ##############################################

with tf.name_scope(loss_tag + "embedding"):
    trans_M = tf.Variable(tf.random_normal([rnn_hidden_dim, rnn_hidden_dim], mean=0, stddev=0.01), name="trans_M")
    tf.summary.histogram('trans_M', trans_M)
    trans_N = tf.Variable(tf.random_normal([user_dim, rnn_hidden_dim], mean=0, stddev=0.01), name="trans_N")
    tf.summary.histogram('trans_N', trans_N)

    user_embedding_matrix = tf.Variable(tf.random_normal([user_size, user_dim], mean=0, stddev=0.01), name="user_embedding_matrix")
    tf.summary.histogram('user_embedding', user_embedding_matrix)

    que_att_v = tf.Variable(tf.random_normal([1, rnn_hidden_dim], stddev=0.01), name="que_att_v")
    tf.summary.histogram('que_att_v', que_att_v)


    word_att_feed_forward_w_word = tf.Variable(tf.random_normal([rnn_hidden_dim, attention_size], stddev=0.01), name="word_att_feed_forward_w_word")
    tf.summary.histogram('word_att_feed_forward_w_word', word_att_feed_forward_w_word)
    word_att_feed_forward_w_que = tf.Variable(tf.random_normal([rnn_hidden_dim, attention_size], stddev=0.01), name="word_att_feed_forward_w_que")
    tf.summary.histogram('word_att_feed_forward_w_que', word_att_feed_forward_w_que)
    word_att_feed_forward_w_user = tf.Variable(tf.random_normal([user_dim, attention_size], stddev=0.01), name="word_att_feed_forward_w_user")
    tf.summary.histogram('word_att_feed_forward_w_user', word_att_feed_forward_w_user)
    word_att_feed_forward_b = tf.Variable(tf.random_normal([attention_size], stddev=0.01), name="word_att_feed_forward_b")
    tf.summary.histogram('word_att_feed_forward_b', word_att_feed_forward_b)
    word_att_feed_forward_v = tf.Variable(tf.random_normal([1, attention_size], stddev=0.01), name="word_att_feed_forward_v")
    tf.summary.histogram('word_att_feed_forward_v', word_att_feed_forward_v)

    sen_att_feed_forward_w_sen = tf.Variable(tf.random_normal([rnn_hidden_dim, attention_size], stddev=0.01), name="sen_att_feed_forward_w_sen")
    tf.summary.histogram('sen_att_feed_forward_w_sen', sen_att_feed_forward_w_sen)
    sen_att_feed_forward_w_que = tf.Variable(tf.random_normal([rnn_hidden_dim, attention_size], stddev=0.01), name="sen_att_feed_forward_w_que")
    tf.summary.histogram('sen_att_feed_forward_w_que', sen_att_feed_forward_w_que)
    sen_att_feed_forward_w_user = tf.Variable(tf.random_normal([user_dim, attention_size], stddev=0.01), name="sen_att_feed_forward_w_user")
    tf.summary.histogram('sen_att_feed_forward_w_user', sen_att_feed_forward_w_user)
    sen_att_feed_forward_b = tf.Variable(tf.random_normal([attention_size], stddev=0.01), name="sen_att_feed_forward_b")
    tf.summary.histogram('sen_att_feed_forward_b', sen_att_feed_forward_b)
    sen_att_feed_forward_v = tf.Variable(tf.random_normal([1, attention_size], stddev=0.01), name="sen_att_feed_forward_v")
    tf.summary.histogram('sen_att_feed_forward_v', sen_att_feed_forward_v)


    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embedding_dim]), trainable=True, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, word_embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    tf.summary.histogram('word_embedding', W)

    sen_zero_padding = tf.zeros(shape=[1,rnn_hidden_dim], dtype=tf.float32)
    word_alpha_zero_padding = tf.zeros(shape=[1], dtype=tf.float32)

############################################## attention ##############################################
def ans_word_level_att(ans_rnn, que_emb, user_emb, ans_sen_len, ans_len, ans_indicies, ans_sen_indicies, ans_sen_padding_indicies):
    que_after_indicies = tf.gather(que_emb, ans_indicies) 
    que_for_sen_att = tf.tensordot(que_after_indicies, word_att_feed_forward_w_que, axes=[[1], [0]])
    que_for_word_att = tf.gather(que_for_sen_att, ans_sen_indicies)

    user_after_indicies = tf.gather(user_emb, ans_indicies)
    user_for_sen_att = tf.tensordot(user_after_indicies, word_att_feed_forward_w_user, axes=[[1], [0]])
    user_for_word_att = tf.gather(user_for_sen_att, ans_sen_indicies)

    ans_word_output = tf.tensordot(ans_rnn, word_att_feed_forward_w_word, axes=[[2], [0]])  # ans_rnn

    ans_word_att = tf.add(tf.add(ans_word_output, que_for_word_att), user_for_word_att)
    ans_word_att = tf.nn.bias_add(ans_word_att, word_att_feed_forward_b)

    ans_word_mask = tf.sequence_mask(ans_sen_len, maxlen=tf.reduce_max(ans_sen_len), dtype=tf.float32)
    ans_word_mask = tf.expand_dims(ans_word_mask, -1)
    ans_word_att = tf.multiply(ans_word_att, ans_word_mask)
    ans_word_att = tf.tanh(ans_word_att)
    ans_word_alpha = tf.tensordot(ans_word_att, word_att_feed_forward_v, axes=[[2], [1]])
    ans_word_alpha = tf.nn.softmax(ans_word_alpha, dim=1)
    ans_weight_matmul = tf.multiply(ans_rnn, ans_word_alpha)    # ans_rnn
    ans_sen_last = tf.reduce_sum(ans_weight_matmul, axis=1)

    ans_sen_mask = tf.sequence_mask(ans_len, maxlen=tf.reduce_max(ans_len), dtype=tf.float32)
    ans_sen_mask = tf.expand_dims(ans_sen_mask, -1)
    ans_sen_last = tf.gather(ans_sen_last, ans_sen_padding_indicies)
    ans_sen_last =  tf.multiply(ans_sen_last, ans_sen_mask)

    return ans_sen_last, ans_word_alpha

def ans_sen_level_att(ans_sen_rnn, que_emb, user_emb, ans_len, ans_indicies, ans_sen_padding_indicies):
    que_after_indicies = tf.gather(que_emb, ans_indicies)
    que_for_sen_att_2 = tf.tensordot(que_after_indicies, sen_att_feed_forward_w_que, axes=[[1], [0]])
    que_for_sen_att_2 = tf.gather(que_for_sen_att_2, ans_sen_padding_indicies)

    user_after_indicies = tf.gather(user_emb, ans_indicies)
    user_for_sen_att_2 = tf.tensordot(user_after_indicies, sen_att_feed_forward_w_user, axes=[[1], [0]])
    user_for_sen_att_2 = tf.gather(user_for_sen_att_2, ans_sen_padding_indicies)

    ans_sen_output = tf.tensordot(ans_sen_rnn, sen_att_feed_forward_w_sen, axes=[[2],[0]])

    ans_sen_att = tf.add(tf.add(ans_sen_output, que_for_sen_att_2), user_for_sen_att_2)
    ans_sen_att = tf.nn.bias_add(ans_sen_att, sen_att_feed_forward_b)

    ans_sen_mask = tf.sequence_mask(ans_len, maxlen=tf.reduce_max(ans_len), dtype=tf.float32)
    ans_sen_mask = tf.expand_dims(ans_sen_mask, -1)
    ans_sen_att = tf.multiply(ans_sen_att, ans_sen_mask)
    ans_sen_att = tf.tanh(ans_sen_att)
    ans_sen_alpha = tf.tensordot(ans_sen_att, sen_att_feed_forward_v, axes=[[2], [1]])
    ans_sen_alpha = tf.nn.softmax(ans_sen_alpha, dim=1)
    ans_sen_weight_matmul = tf.multiply(ans_sen_rnn, ans_sen_alpha)
    ans_last = tf.reduce_sum(ans_sen_weight_matmul, axis=1)

    return ans_last, ans_sen_alpha
############################################## RNN ##############################################

def ans_word_RNN(ans_words_vec, seqlen):
    with tf.variable_scope(loss_tag + "ans_word_RNN"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_dim, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(tf.shape(ans_words_vec)[0], dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, ans_words_vec, sequence_length=seqlen,
                                                 initial_state=init_state, time_major=False)
        return outputs

def ans_sen_RNN(ans_sen_vec, ans_len):
    with tf.variable_scope(loss_tag + "ans_sen_RNN"):
        # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_dim, forget_bias=1.0, state_is_tuple=True)
        # else:
        #     lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_hidden_dim)
        init_state = lstm_cell.zero_state(tf.shape(ans_sen_vec)[0], dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, ans_sen_vec, sequence_length=ans_len,
                                                 time_major=False, initial_state=init_state)#, dtype=tf.float32)
        return outputs

############################################## main logic ##############################################
with tf.variable_scope(loss_tag + "my_word_rnn") as scope:
    que_words_vec = tf.nn.embedding_lookup(W, place_que_words_index)
    que_rnn = ans_word_RNN(que_words_vec, place_que_sen_len)

    que_sequence_mask = tf.sequence_mask(place_que_sen_len, maxlen=tf.reduce_max(place_que_sen_len), dtype=tf.float32)
    que_sequence_mask = tf.expand_dims(que_sequence_mask, -1)
    que_alpha = tf.tensordot(que_rnn, que_att_v, axes=[[2], [1]])   #que_rnn
    que_alpha = tf.nn.softmax(que_alpha, dim=1, name="que_alpha")
    que_weight_matmul = tf.multiply(que_rnn, que_alpha) # que_rnn
    que_last = tf.reduce_sum(que_weight_matmul, axis=1)

    scope.reuse_variables()
    ####### positive
    positive_ans_words_vec = tf.nn.embedding_lookup(W, place_positive_ans_words_index)
    positive_ans_sen_rnn = ans_word_RNN(positive_ans_words_vec, place_positive_ans_sen_len)
    positive_user_emb = tf.nn.embedding_lookup(user_embedding_matrix, place_positive_user_lookup)
    positive_ans_sen_emb, positive_ans_word_att_alpha = ans_word_level_att(positive_ans_sen_rnn, que_last, positive_user_emb, place_positive_ans_sen_len,
                                                                               place_positive_ans_len, place_positive_ans_indicies,
                                                                               place_positive_ans_sen_indicies, place_positive_ans_sen_padding_indicies)
    positive_ans_word_att_alpha = tf.reshape(positive_ans_word_att_alpha, [tf.shape(place_positive_ans_sen_len)[0], tf.reduce_max(place_positive_ans_sen_len), 1], name="positive_ans_word_att_alpha")

    ####### negative
    negative_ans_words_vec = tf.nn.embedding_lookup(W, place_negative_ans_words_index)
    negative_ans_sen_rnn = ans_word_RNN(negative_ans_words_vec, place_negative_ans_sen_len)
    negative_user_emb = tf.nn.embedding_lookup(user_embedding_matrix, place_negative_user_lookup)
    negative_ans_sen_emb, negative_ans_word_att_alpha = ans_word_level_att(negative_ans_sen_rnn, que_last, negative_user_emb, place_negative_ans_sen_len,
                                                                               place_negative_ans_len, place_negative_ans_indicies,
                                                                               place_negative_ans_sen_indicies, place_negative_ans_sen_padding_indicies)
    negative_ans_word_att_alpha = tf.reshape(negative_ans_word_att_alpha, [tf.shape(place_negative_ans_sen_len)[0], tf.reduce_max(place_negative_ans_sen_len), 1], name="negative_ans_word_att_alpha")
with tf.variable_scope(loss_tag + "my_sen_rnn") as scope:
    ####### positive
    positive_ans_rnn = ans_sen_RNN(positive_ans_sen_emb, place_positive_ans_len)
    positive_ans_emb, positive_ans_sen_att_alpha = ans_sen_level_att(positive_ans_rnn, que_last, positive_user_emb, place_positive_ans_len,
                                                                     place_positive_ans_indicies, place_positive_ans_sen_padding_indicies)
    positive_ans_sen_att_alpha = tf.reshape(positive_ans_sen_att_alpha, [batch_pair_size, tf.reduce_max(place_positive_ans_len), 1], name="positive_ans_sen_att_alpha")
    positive_qa_score = tf.reduce_sum(tf.multiply(tf.matmul(que_last, trans_M), positive_ans_emb), axis=1)
    positive_qu_score = tf.reduce_sum(tf.multiply(tf.matmul(positive_user_emb, trans_N), que_last), axis=1)
    positive_score = tf.add(positive_qa_score, positive_qu_score, name="positive_score")
    ####### negative
    scope.reuse_variables()
    negative_ans_rnn = ans_sen_RNN(negative_ans_sen_emb, place_negative_ans_len)
    negative_ans_emb, negative_ans_sen_att_alpha = ans_sen_level_att(negative_ans_rnn, que_last, negative_user_emb, place_negative_ans_len,
                                                                     place_negative_ans_indicies, place_negative_ans_sen_padding_indicies)
    negative_ans_sen_att_alpha = tf.reshape(negative_ans_sen_att_alpha, [batch_pair_size, tf.reduce_max(place_negative_ans_len), 1], name="negative_ans_sen_att_alpha")
    negative_qa_score = tf.reduce_sum(tf.multiply(tf.matmul(que_last, trans_M), negative_ans_emb), axis=1)
    negative_qu_score = tf.reduce_sum(tf.multiply(tf.matmul(negative_user_emb, trans_N), que_last), axis=1)
    negative_score = tf.add(negative_qa_score, negative_qu_score, name="negative_score")


# hinge_loss
hinge_loss = tf.reduce_sum(tf.nn.relu(margin + negative_score - positive_score), name=loss_tag + "hinge_loss")
l2_loss = tf.nn.l2_loss(positive_user_emb)
l2_loss += tf.nn.l2_loss(negative_user_emb)
loss = hinge_loss + alpha * l2_loss
tf.summary.scalar('hinge_loss', hinge_loss)
tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name=loss_tag + "optimizer").minimize(loss, aggregation_method=1)

# acc
acc_count = tf.count_nonzero(tf.greater(positive_score, negative_score))



# Initializing the variables
init = tf.global_variables_initializer()



def getIndicies(ans_len_list, ans_sen_len_list):
    max_ans_sen_len = max(ans_sen_len_list)
    ans_indicies = []
    for i in range(ans_len_list.__len__()):
        this_ans_len = ans_len_list[i]
        for j in range(this_ans_len):
            ans_indicies.append(i)
    ans_sen_indices = []
    for i in range(ans_indicies.__len__()):
        temp = []
        for j in range(max_ans_sen_len):
            temp.append(i)
        ans_sen_indices.append(temp)

    ans_sen_padding_indicies = []
    max_ans_len = max(ans_len_list)
    end = 0
    for i in range(ans_len_list.__len__()):
        start = end
        this_ans_len = ans_len_list[i]
        end = start + this_ans_len

        temp=[]
        for j in range(start, end):
            temp.append(j)
        for j in range(this_ans_len, max_ans_len):
            temp.append(0)
        ans_sen_padding_indicies.append(temp)

    return ans_indicies, ans_sen_indices, ans_sen_padding_indicies



config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config = config) as sess:
# with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(init)



    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    saver = tf.train.Saver()

    train_loops = int(my_dataset.train_size *10/batch_pair_size)
    val_loops = int(my_dataset.val_size/batch_pair_size)+1
    val_for = int(my_dataset.val_size *0.5/batch_pair_size)


    print_for = 500
    log_for = 20
    my_evaluation = evaluation()

    all_cost = 0.0
    max_precision = 0.0
    # all_len = []
    for train_step in range(train_loops):
        try:
            # print("************* ", time.time(), " *************")
            all_len = []
            positive_ans_sen_words, positive_ans_sen_len, positive_ans_len, positive_user_id, \
            negative_ans_sen_words, negative_ans_sen_len, negative_ans_len, negative_user_id, \
            que_sen_words, que_sen_len = my_dataset.next_batch_train(batch_size=batch_pair_size, isTrain=True)

            positive_ans_indicies, positive_ans_sen_indicies, positive_ans_sen_padding_indicies = getIndicies(positive_ans_len, positive_ans_sen_len)
            negative_ans_indicies, negative_ans_sen_indicies, negative_ans_sen_padding_indicies = getIndicies(negative_ans_len, negative_ans_sen_len)

            loss_, _, q_alpha, p_alpha, n_alpha= sess.run([loss, optimizer, que_alpha, positive_ans_word_att_alpha, negative_ans_sen_att_alpha],
                                                   feed_dict={place_que_words_index:que_sen_words,
                                                        place_que_sen_len:que_sen_len,
                                                        place_positive_ans_words_index:positive_ans_sen_words,
                                                        place_positive_ans_sen_len:positive_ans_sen_len,
                                                        place_positive_ans_len:positive_ans_len,
                                                        place_positive_user_lookup:positive_user_id,
                                                        place_negative_ans_words_index:negative_ans_sen_words,
                                                        place_negative_ans_sen_len:negative_ans_sen_len,
                                                        place_negative_ans_len:negative_ans_len,
                                                        place_negative_user_lookup:negative_user_id,

                                                        place_positive_ans_indicies:positive_ans_indicies,
                                                        place_positive_ans_sen_indicies:positive_ans_sen_indicies,
                                                        place_positive_ans_sen_padding_indicies:positive_ans_sen_padding_indicies,
                                                        place_negative_ans_indicies:negative_ans_indicies,
                                                        place_negative_ans_sen_indicies:negative_ans_sen_indicies,
                                                        place_negative_ans_sen_padding_indicies:negative_ans_sen_padding_indicies})
            all_cost += loss_
            all_len.append(max(positive_ans_sen_len))
            all_len.append(max(negative_ans_sen_len))
            # logging.info("train_step: "+ str(train_step)+ "  loss: "+ str(loss_)+ " max_len: "+ str(max(all_len)))

            if train_step%log_for == 0:
                _,record = sess.run([optimizer,merged],feed_dict={place_que_words_index:que_sen_words,
                                                        place_que_sen_len:que_sen_len,
                                                        place_positive_ans_words_index:positive_ans_sen_words,
                                                        place_positive_ans_sen_len:positive_ans_sen_len,
                                                        place_positive_ans_len:positive_ans_len,
                                                        place_positive_user_lookup:positive_user_id,
                                                        place_negative_ans_words_index:negative_ans_sen_words,
                                                        place_negative_ans_sen_len:negative_ans_sen_len,
                                                        place_negative_ans_len:negative_ans_len,
                                                        place_negative_user_lookup:negative_user_id,

                                                        place_positive_ans_indicies:positive_ans_indicies,
                                                        place_positive_ans_sen_indicies:positive_ans_sen_indicies,
                                                        place_positive_ans_sen_padding_indicies:positive_ans_sen_padding_indicies,
                                                        place_negative_ans_indicies:negative_ans_indicies,
                                                        place_negative_ans_sen_indicies:negative_ans_sen_indicies,
                                                        place_negative_ans_sen_padding_indicies:negative_ans_sen_padding_indicies})

                writer.add_summary(record, train_step*batch_pair_size)

            if train_step%print_for == (print_for-1):
                print_size = batch_pair_size*print_for
                all_cost /= print_size
                # print("train_step: ", train_step, "  loss: ", all_cost, "max_len: ", max(all_len))
                logging.info("train_step: "+str(train_step)+ "  loss: "+str(all_cost)+"max_len: "+str(max(all_len)))
                all_cost = 0.0
                all_len = []
                # print("q_alpha: ", str(q_alpha), str(q_alpha.shape))
                # print("p_alpha: ", str(p_alpha), str(p_alpha.shape))
                # print("n_alpha: ", str(n_alpha), str(n_alpha.shape))

            ########### save ###########

            ###########  ###########
            if train_step%val_for == (val_for-1):
                # print("******** val ********")
                logging.info("******** val ********")
                time1 = time.time()
                val_loss = 0.0
                count = 0.0
                val_score_dict = dict()
                for val_step in range(val_loops):
                    # testset next
                    positive_ans_sen_words, positive_ans_sen_len, positive_ans_len, positive_user_id, \
                    negative_ans_sen_words, negative_ans_sen_len, negative_ans_len, negative_user_id, \
                    que_sen_words, que_sen_len, positive_id_list, negative_id_list = my_dataset.next_batch_test(batch_size=batch_pair_size, isVal=True)

                    positive_ans_indicies, positive_ans_sen_indicies, positive_ans_sen_padding_indicies = getIndicies(positive_ans_len, positive_ans_sen_len)
                    negative_ans_indicies, negative_ans_sen_indicies, negative_ans_sen_padding_indicies = getIndicies(negative_ans_len, negative_ans_sen_len)


                    acc_count_, val_loss_step, positive_score_, negative_score_  = sess.run([acc_count, loss, positive_score, negative_score],
                                                         feed_dict={place_que_words_index:que_sen_words,
                                                                    place_que_sen_len:que_sen_len,
                                                                    place_positive_ans_words_index:positive_ans_sen_words,
                                                                    place_positive_ans_sen_len:positive_ans_sen_len,
                                                                    place_positive_ans_len:positive_ans_len,
                                                                    place_positive_user_lookup:positive_user_id,
                                                                    place_negative_ans_words_index:negative_ans_sen_words,
                                                                    place_negative_ans_sen_len:negative_ans_sen_len,
                                                                    place_negative_ans_len:negative_ans_len,
                                                                    place_negative_user_lookup:negative_user_id,

                                                                    place_positive_ans_indicies:positive_ans_indicies,
                                                                    place_positive_ans_sen_indicies:positive_ans_sen_indicies,
                                                                    place_positive_ans_sen_padding_indicies:positive_ans_sen_padding_indicies,
                                                                    place_negative_ans_indicies:negative_ans_indicies,
                                                                    place_negative_ans_sen_indicies:negative_ans_sen_indicies,
                                                                    place_negative_ans_sen_padding_indicies:negative_ans_sen_padding_indicies})
                    for i in range(positive_id_list.__len__()):
                        val_score_dict[str(positive_id_list[i])] = positive_score_[i]
                        val_score_dict[str(negative_id_list[i])] = negative_score_[i]
                    val_loss += val_loss_step
                    count += acc_count_
                val_size = batch_pair_size * val_loops
                accuracy = count / val_size
                val_loss /= val_size
                time2 = time.time() - time1
                # print("train_step: ", train_step, " loss:", val_loss, " accuracy: "+ accuracy , "time: "+time2)
                logging.info("train_step: "+ str(train_step)+ " loss:"+str(val_loss)+" accuracy: "+ str(accuracy) + "time: "+str(time2))
                precision_1, accuray_1, all_DCG, MAP, MRR  = my_evaluation.evaluate(val_score_dict)
                logging.info(str(precision_1) + " "+ str(accuray_1)+" "+str(all_DCG)+" "+str(MAP)+" "+str(MRR))
                if precision_1>max_precision:
                    max_precision = precision_1
                    logging.info("******** save ********")
                    save_path = saver.save(sess, save_path=save_dir, global_step=train_step)
        except Exception as err:
            # print(str(err))
            logging.warning(str(err))
            break
