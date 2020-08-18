import tensorflow as tf
import random
import numpy as np

class BILSTM(object):

    def __init__(self,embedding_dim,hidden_dim,vocab_size_char,vocab_size_bio,vocab_size_attr,tag_indx,use_crf):

        self.inputs_seq=tf.placeholder(tf.int32,[None,None],name="input_seq")
        self.inputs_seq_len=tf.placeholder(tf.int32,[None],name="inputs_seq_len")
        self.outputs_seq_bio=tf.placeholder(tf.int32,[None,None],name="outputs_seq_bio")
        self.outputs_seq_attr=tf.placeholder(tf.int32,[None,None],name="outputs_seq_attr")

        with tf.variable_scope("embedding_layer"):
            embedding_matrix=tf.get_variable("embedding_matrix",[vocab_size_char,embedding_dim],dtype=tf.float32)
            embedded=tf.nn.embedding_lookup(embedding_matrix,self.inputs_seq)

        with tf.variable_scope("encoder"):
            cell_fw=tf.nn.rnn_cell.LSTMCell(hidden_dim)
            cell_bw=tf.nn.rnn_cell.LSTMCell(hidden_dim)
            ((rnn_fw_outputs,rnn_bw_outputs),(rnn_fw_final_state,rnn_bw_final_state))=tf.nn.bidirectional_dynamic_rnn(
                cell_bw=cell_bw,
                cell_fw=cell_fw,
                inputs=embedded,
                sequence_length=self.inputs_seq_len,
                dtype=tf.float32
            )
            rnn_outputs=tf.add(rnn_fw_outputs,rnn_bw_outputs)

        with tf.variable_scope("bio_projection"):
            logits_bio=tf.layers.dense(rnn_outputs,vocab_size_bio)
            probos_bio=tf.nn.softmax(logits_bio,axis=-1)

            if not use_crf:
                preds_bio=tf.argmax(probos_bio,axis=-1,name="preds_bio")
            else:
                log_likelihood,transition_matrix=tf.contrib.crf.crf_log_likelihood(logits_bio,
                                                                                   self.outputs_seq_bio,
                                                                                   self.inputs_seq_len)
                preds_bio,crf_score=tf.contrib.crf.crf_decode(logits_bio,transition_matrix,self.inputs_seq_len)


        with tf.variable_scope("attr_projection"):
            logits_attr=tf.layers.dense(rnn_outputs,vocab_size_attr)
            probs_attr=tf.nn.softmax(logits_attr,axis=-1)
            preds_attr=tf.argmax(probs_attr,axis=-1,name="preds_attr")
        self.outputs=(preds_bio,preds_attr)

        with tf.variable_scope("loss"):
            if not use_crf:
                loss_bio=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bio,labels=self.outputs_seq_bio)
                masks_bio=tf.sequence_mask(self.inputs_seq_len,dtype=tf.float32)
                loss_bio=tf.reduce_sum(loss_bio*masks_bio,axis=-1)

            else:
                loss_bio=-log_likelihood/tf.cast(self.inputs_seq_len,tf.float32)

            loss_attr=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_attr,labels=self.outputs_seq_attr)
            masks_attr = tf.cast(tf.not_equal(preds_bio,tag_indx),tf.float32)

            loss_attr = tf.reduce_sum(loss_attr * masks_attr, axis=-1)

            loss=loss_attr+loss_bio

        self.loss=tf.reduce_mean(loss)

        with tf.variable_scope("opt"):
            self.train_op=tf.train.AdamOptimizer().minimize(loss)

BILSTM(128,32,23,32,23232,1,True)


