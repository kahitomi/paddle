#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import paddle.fluid.layers.control_flow as cf
import paddle.fluid.layers as pd
from paddle.fluid.contrib.decoder.beam_search_decoder import *

class net():
    """
    model
    """
    def __init__(self,
            embedding_dim, 
            encoder_size, 
            decoder_size, 
            source_dict_dim,
            target_dict_dim, 
            tag_dict_dim, 
            is_generating, 
            beam_size, 
            max_length,
            source_entity_dim,
            source_pos_dim,
            embedding_entity_dim,
            embedding_pos_dim,
            end_id):
        # The encoding process. Encodes the input words into tensors.
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.embedding_dim = embedding_dim
        self.source_dict_dim = target_dict_dim
        self.is_generating = is_generating
        self.source_dict_dim = source_dict_dim
        self.target_dict_dim = target_dict_dim
        self.tag_dict_dim = tag_dict_dim
        self.max_length = max_length
        self.end_id = end_id
        self.beam_size = beam_size
        self.no_grad_set = []
        
        self.dropout_prob = 0.5

        src_word_idx = fluid.layers.data(
            name='source_sequence', shape=[1], dtype='int64', lod_level=1)
        # print(src_word_idx.shape)
        self.src_word_idx = src_word_idx

        src_embedding = fluid.layers.embedding(
            input=src_word_idx,
            size=[source_dict_dim, embedding_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(name='emb'))

        src_entity_idx = fluid.layers.data(
            name='source_entities', shape=[1], dtype='int64', lod_level=1)

        entity_embedding = fluid.layers.embedding(
            input=src_entity_idx,
            size=[source_entity_dim, embedding_entity_dim],
            dtype='float32')

        src_pos_idx = fluid.layers.data(
            name='source_pos', shape=[1], dtype='int64', lod_level=1)

        pos_embedding = fluid.layers.embedding(
            input=src_pos_idx,
            size=[source_pos_dim, embedding_pos_dim],
            dtype='float32')
        # print(src_embedding)
        # print(entity_embedding)
        # print(pos_embedding)

        embeddings = fluid.layers.concat(
            input=[src_embedding, entity_embedding, pos_embedding], axis=1)
        # print(embeddings)
        # if not is_generating:
        #     embeddings = fluid.layers.dropout(
        #                     embeddings, dropout_prob=self.dropout_prob)

        src_forward, src_reversed = self.bi_lstm_encoder(
            input_seq=embeddings, gate_size=encoder_size)


        encoded_vector = fluid.layers.concat(
            input=[src_forward, src_reversed], axis=1)
        
        pad_zero = pd.fill_constant(shape=[self.encoder_size * 2], 
                        dtype='float32', value=0)
        encoded_vector_full, encoded_vector_length = pd.sequence_pad(encoded_vector, pad_zero,
                maxlen=self.max_length, name="copy_score_padding")
        print(encoded_vector_full)

        # if not is_generating:
        #     encoded_vector = fluid.layers.dropout(
        #                     encoded_vector, dropout_prob=self.dropout_prob)
        self.encoder_vec = encoded_vector
        self.encoder_vec_full = encoded_vector_full

        encoded_proj = fluid.layers.fc(input=encoded_vector,
                                       size=decoder_size,
                                       bias_attr=False)
        self.encoder_proj = encoded_proj

        backward_first = fluid.layers.sequence_pool(
            input=src_reversed, pool_type='first')
        decoder_boot = fluid.layers.fc(input=backward_first,
                                       size=decoder_size,
                                       bias_attr=False,
                                       act='tanh')
        cell_init = fluid.layers.fill_constant_batch_size_like(
            input=decoder_boot,
            value=1.0,
            shape=[-1, decoder_size],
            dtype='float32')
        # cell_init.stop_gradient = False
        cell_init.stop_gradient = True

        # Create a RNN state cell by providing the input and hidden states, and
        # specifies the hidden state as output.
        # h = InitState(init=decoder_boot, need_reorder=True)
        self.h = decoder_boot
        self.c = cell_init

        event_cla_id = fluid.layers.data(
            name='event_class', shape=[1], dtype='int64')

        self.event_embedding = fluid.layers.embedding(
            input=event_cla_id,
            size=[self.tag_dict_dim, embedding_entity_dim],
            dtype='float32')
        
        # self.decoder_lstm = fluid.contrib.layers.BasicLSTMUnit(
        #     "decoder_lstm",
        #     self.decoder_size,
        #     fluid.ParamAttr(initializer=fluid.initializer.UniformInitializer(
        #         low=-0.1, high=0.1)),
        #     fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0)), )

        #####
        # DECODER
        #####
        label = fluid.layers.data(
            name='label_sequence', shape=[1], dtype='int64', lod_level=1)
        if not is_generating:
            rnn_out = self.train_decoder(decoder_boot)	
            predict_label = fluid.layers.argmax(x=rnn_out, axis=1)
            # print(label.shape)
            # print(rnn_out.shape)
            # print(predict_label.shape)
            cost = fluid.layers.cross_entropy(input=rnn_out, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            self.predict = rnn_out
            self.label = predict_label
            self.avg_cost = avg_cost
            feeding_list = ["source_sequence", "source_entities", 
                    "source_pos", "event_class", "source_index", "target_sequence", "label_sequence"]
            # return avg_cost, feeding_list

            self.feeding_list = feeding_list
        else:
            # rnn_out = self.train_decoder(decoder_boot)	
            # translation_ids = fluid.layers.argmax(x=rnn_out, axis=-1)

            beam_search_out = self.decoder(decoder_boot)
            translation_ids, translation_scores = beam_search_out
            feeding_list = ["source_sequence", "source_entities", 
                    "source_pos", "event_class", "source_index", "label_sequence"]
            # feeding_list = ["source_sequence", "source_entities", 
            #         "source_pos", "target_sequence"]
            # feeding_list = ["source_sequence", "source_entities", 
            #         "source_pos", "target_sequence", "label_sequence"]

            # return translation_ids, translation_scores, feeding_list
            self.translation_ids = translation_ids
            self.translation_scores = translation_scores
            self.feeding_list = feeding_list

        self.no_grad_set = set(self.no_grad_set)

    def train_decoder(self, boot_state):
        """
        # Define input data of sequence id of target language and reflect it on word vector of low-dimension language space
        """
        trg_word_idx = fluid.layers.data(
            name='target_sequence', shape=[1], dtype='int64', lod_level=1)
        trg_embedding = fluid.layers.embedding(
            input=trg_word_idx,
            size=[self.target_dict_dim, self.embedding_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(name="trg_embedding"))
        # last_ids = fluid.layers.fill_constant_batch_size_like(
        #                 input=boot_state,
        #                 value=0,
        #                 shape=[-1, 1],
        #                 dtype='int64')
        # last_ids = fluid.layers.sequence_expand_as(last_ids, self.src_word_idx)
        prob_c = fluid.layers.fill_constant_batch_size_like(
                        input=boot_state,
                        value=1.0,
                        shape=[-1, 1],
                        dtype='float32')
        # print(prob_c)
        src_word_idx_float32 = fluid.layers.cast(self.src_word_idx, "float32")
        # print(boot_state)
        # print(src_word_idx_float32)
        prob_c = fluid.layers.sequence_expand_as(prob_c,
                src_word_idx_float32)
        prob_c.stop_gradient = True
        # print(prob_c)
        # prob_c = fluid.layers.sequence_expand(prob_c,
        #         src_word_idx_float32,
        #         ref_level=-1)
        # print(prob_c)

        # input_id_vec = fluid.layers.fill_constant_batch_size_like(
        #                 input=boot_state,
        #                 value=0.0,
        #                 shape=[-1, self.target_dict_dim],
        #                 dtype='float32')

        # input_id_vec = pd.one_hot(
        #         self.src_word_idx, depth=self.target_dict_dim)

        # src_indexes = pd.range(0, self.max_length, 1, "int64")
        # src_indexes = self.src_word_idx
        src_indexes = fluid.layers.data(
            name='source_index', shape=[1], dtype='int64', lod_level=1)
        # print(src_indexes)

        
        _param_c = fluid.ParamAttr(name="input_context_c_w")
        _bias_c = fluid.ParamAttr(name="input_context_c_b")
        _param_h = fluid.ParamAttr(name="input_context_h_w")
        _bias_h = fluid.ParamAttr(name="input_context_h_b")
        _param_s = fluid.ParamAttr(name="out_softmax_w")
        _bias_s = fluid.ParamAttr(name="out_softmax_b")

        # _param_c = fluid.initializer.Normal(loc=0.0, scale=2.0)
        # _bias_c = fluid.initializer.Normal(loc=0.0, scale=2.0)
        # _param_h = fluid.initializer.Normal(loc=0.0, scale=2.0)
        # _bias_h = fluid.initializer.Normal(loc=0.0, scale=2.0)
        # _param_s = fluid.initializer.Normal(loc=0.0, scale=2.0)
        # _bias_s = fluid.initializer.Normal(loc=0.0, scale=2.0)

        rnn = fluid.layers.DynamicRNN()
        # pre_h_state = rnn.memory(init=self.h, need_reorder=True)
        # pre_c_state = rnn.memory(init=self.c)
        with rnn.block(): # use DynamicRNN to define computation at each step
            # Fetch input word vector of target language at present step
            current_word = rnn.step_input(trg_embedding)
            _last_ids = rnn.step_input(trg_word_idx)
            # pd.Print(trg_embedding)

            encoder_vec = rnn.static_input(self.encoder_vec)
            print(encoder_vec)
            encoder_vec_full = rnn.static_input(self.encoder_vec_full)
            print(encoder_vec_full)
            encoder_proj = rnn.static_input(self.encoder_proj)
            # _encoder_input_ids = rnn.static_input(self.src_word_idx)
            _encoder_input_ids = rnn.static_input(src_indexes)
            _prob_c = rnn.static_input(prob_c)
            # _input_id_vec = rnn.static_input(input_id_vec)

            event_embedding = rnn.static_input(self.event_embedding)

            # obtain state of hidden layer
            pre_h_state = rnn.memory(init=self.h, need_reorder=True)
            # pre_h_state = rnn.memory(init=self.h)

            pre_c_state = rnn.memory(init=self.c, need_reorder=True)
            # pre_c_state = rnn.memory(init=self.c)

            # print(self.encoder_vec)
            # print(self.encoder_proj)
            # print("pre_h_state", pre_h_state)

            # ATTENTION
            # pd.Print(pre_c_state)
            att_context = self.simple_attention(
                    encoder_vec, encoder_proj, pre_h_state)
            # pd.Print(att_context, summarize=10)

            # print("decoder_inputs", decoder_inputs)

            # # computing unit of decoder
            # current_state = fluid.layers.fc(input=[current_word, pre_state, att_context],
            #                       size=self.decoder_size,
            #                       act='tanh')




            # print(current_word)
            # # print(att_context)
            # # input_context = pd.concat([current_word, att_context], axis=1)
            # input_context = current_word
            # print(input_context)
            # input_context = pd.concat([input_context, pre_h_state, pre_c_state], axis=1)
            # # input_context_h = pd.concat([input_context, pre_h_state, pre_c_state], axis=1)
            # current_h = fluid.layers.fc(input=input_context, 
            #                 size=self.decoder_size,
            #                 act='tanh',
            #                 param_attr=_param_h,
            #                 bias_attr=_bias_h)
            # # input_context_c = pd.concat([input_context, pre_c_state], axis=1)
            # # print(input_context_c)
            # # print(self.decoder_size)
            # # # current_c = current_h
            # # current_c = fluid.layers.fc(input=current_h, 
            # #                 # size=self.target_dict_dim,
            # #                 size=self.decoder_size,
            # #                 # act='relu',
            # #                 act='tanh',
            # #                 # act='softmax',
            # #                 param_attr=_param_c,
            # #                 bias_attr=_bias_c)
            # current_c = fluid.layers.fc(input=input_context, 
            #                 # size=self.target_dict_dim,
            #                 size=self.decoder_size,
            #                 # act='relu',
            #                 act='tanh',
            #                 # act='softmax',
            #                 param_attr=_param_c,
            #                 bias_attr=_bias_c)

            # # pd.Print(current_word)
            # # pd.Print(att_context)
            # last_ids_add_dim = _last_ids
            # expand_last_ids = fluid.layers.sequence_expand_as(_last_ids, _encoder_input_ids)
            # expand_last_ids.stop_gradient = True
            # if_last_word_in_input = fluid.layers.equal(_encoder_input_ids, expand_last_ids)
            # if_last_word_in_input.stop_gradient = True
            # if_last_word_in_input = fluid.layers.cast(if_last_word_in_input, "float32")
            # mask_encoder_vec = fluid.layers.elementwise_mul(encoder_vec, if_last_word_in_input, axis=0)
            # weighted_encoder_vec = fluid.layers.elementwise_mul(mask_encoder_vec, _prob_c, axis=0)
            # selective_read = fluid.layers.sequence_pool(weighted_encoder_vec, pool_type='max')

            # current_word = fluid.layers.concat([current_word, att_context, 
            #     selective_read, event_embedding], axis=1)
            # # current_word = fluid.layers.fc(input=current_word, 
            # #                 size=self.decoder_size,
            # #                 param_attr=_param_c,
            # #                 bias_attr=False)

            
            # # pd.Print(current_word)
            # # pd.Print(pre_h_state)
            # # pd.Print(pre_c_state)
            # current_h, current_c = self.decoder_lstm(current_word, pre_h_state, pre_c_state)
            
            # pd.Print(encoder_vec, summarize=10)
            current_score, current_h, current_c, this_prob_c = self.copy_decoder(
                current_word, 
                encoder_vec, encoder_vec_full, encoder_proj, 
                _encoder_input_ids, _last_ids,
                _prob_c, att_context,
                pre_h_state, pre_c_state,
                event_embedding)

            # current_h, current_c = pd.lstm_unit(
            #         x_t=input_context, 
            #         hidden_t_prev=pre_h_state,
            #         cell_t_prev=pre_c_state,
            #         name="decoder_lstm")

            # current_score = fluid.layers.fc(input=input_context_c,
            #                       size=self.target_dict_dim,
            #                       act='softmax',
            #                       param_attr=_param_s,
            #                       bias_attr=_bias_s)

            # input_context_s = pd.concat([current_h, current_c], axis=1)
            # current_score = fluid.layers.fc(input=input_context_s,
            # current_score = fluid.layers.fc(input=current_h,
            #                       size=self.target_dict_dim,
            #                       act='softmax',
            #                       # act='tanh',
            #                       param_attr=_param_s,
            #                       bias_attr=_bias_s)
            # pd.Print(current_score)

            # current_score = fluid.layers.fc(input=[current_word, att_context], 
            #                 size=self.decoder_size, act='relu')
            # this_prob_c = fluid.layers.fc(input=[current_word, att_context], 
            #                 size=self.decoder_size, act='relu')


            # pd.Print(current_h, summarize=10)
            # pd.Print(current_c, summarize=10)
            # pd.Print(this_prob_c, summarize=10)
            # pd.Print(current_score, summarize=10)

            # decoder_inputs = fluid.layers.concat(
            #     input=[att_context, current_word, copy_out], axis=1)
            # # current_h, current_c = pd.lstm_unit(decoder_inputs, 
            # #         pre_h_state, pre_c_state, self.max_length, 
            # #         self.decoder_size, 1)
            # # compute predicting probability of nomarlized word
            # current_h, current_c = self.lstm_step(
            #         decoder_inputs, pre_h_state, pre_c_state, self.decoder_size)
            # current_score = fluid.layers.fc(input=current_h,
            #                       size=self.target_dict_dim,
            #                       act='softmax',
            #                       param_attr=fluid.ParamAttr(name="out_softmax_w"),
            #                       bias_attr=fluid.ParamAttr(name="out_softmax_b"))
            # update hidden layer of RNN
            # rnn.update_memory(pre_state, current_state)
            rnn.update_memory(pre_h_state, current_h)
            rnn.update_memory(pre_c_state, current_c)
            # output predicted probability
            rnn.output(current_score)
        # print(rnn.mem_dict)
        # print(rnn.mem_link)
        return rnn()

    def decoder(self, init_state):
        """
        implement decoder in inference mode
        """
        # pd.Print(init_state)
        # define counter variable in the decoding
        array_len = pd.fill_constant(shape=[1], dtype='int64', value=self.max_length)
        counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)
        static_count = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

        # define tensor array to save content at each time step, and write initial id, score and state
        state_h_array = pd.create_array('float32')
        pd.array_write(self.h, array=state_h_array, i=counter)
        state_c_array = pd.create_array('float32')
        pd.array_write(self.c, array=state_c_array, i=counter)

        src_indexes = fluid.layers.data(
            name='source_index', shape=[1], dtype='int64', lod_level=1)
        src_index_array = pd.create_array('int64')
        pd.array_write(src_indexes, array=src_index_array, i=counter)

        ids_array = pd.create_array('int64')
        scores_array = pd.create_array('float32')

        init_ids = fluid.layers.data(
            name="init_ids", shape=[1], dtype="int64", lod_level=2)
        init_scores = fluid.layers.data(
            name="init_scores", shape=[1], dtype="float32", lod_level=2)

        pd.array_write(init_ids, array=ids_array, i=counter)
        pd.array_write(init_scores, array=scores_array, i=counter)

        encoder_vec_array = pd.create_array('float32')
        pd.array_write(self.encoder_vec, array=encoder_vec_array, i=static_count)
        encoder_vec_full_array = pd.create_array('float32')
        pd.array_write(self.encoder_vec_full, array=encoder_vec_full_array, i=static_count)
        encoder_proj_array = pd.create_array('float32')
        pd.array_write(self.encoder_proj, array=encoder_proj_array, i=static_count)

        event_embedding_array = pd.create_array('float32')
        pd.array_write(self.event_embedding, array=event_embedding_array, i=static_count)

        # define conditional variable to stop loop
        cond = pd.less_than(x=counter, y=array_len)
        # define while_op
        while_op = pd.While(cond=cond)
        with while_op.block(): # define the computing of each step
            # pd.Print(counter)

            # obtain input at present step of decoder, including id chosen at previous step, corresponding score and state at previous step.
            pre_ids = pd.array_read(array=ids_array, i=counter)
            pre_h_state = pd.array_read(array=state_h_array, i=counter)
            pre_c_state = pd.array_read(array=state_c_array, i=counter)

            # pre_score = pd.array_read(array=scores_array, i=counter)
            pre_score = pd.array_read(array=scores_array, i=static_count)

            _encoder_input_ids = pd.array_read(
                    array=src_index_array, i=static_count)

            event_embedding = pd.array_read(
                    array=event_embedding_array, i=static_count)

            # print("pre_h_state", pre_h_state)
            encoder_vec = pd.array_read(
                    array=encoder_vec_array, i=static_count)
            encoder_vec_full = pd.array_read(
                    array=encoder_vec_full_array, i=static_count)
            encoder_proj = pd.array_read(
                    array=encoder_proj_array, i=static_count)

            # # update input state as state correspondent with id chosen at previous step
            # pre_h_state_expanded = pd.sequence_expand(pre_h_state, pre_score)
            # pre_c_state_expanded = pd.sequence_expand(pre_c_state, pre_score)
            # computing logic of decoder under the same train mode, including input vector and computing unit of decoder
            # compute predicting probability of normalized word
            pre_ids_emb = pd.embedding(
                input=pre_ids,
                size=[self.target_dict_dim, self.embedding_dim],
                dtype='float32',
                param_attr=fluid.ParamAttr(name="trg_embedding"))

            # pd.Print(pre_ids_emb)
            att_context = self.simple_attention(
                    encoder_vec, encoder_proj, pre_h_state)
            # print("att_context", att_context)
            # print("pre_ids_emb", pre_ids_emb)
            # pd.Print(att_context)

            prob_c = fluid.layers.sequence_expand_as(pre_score,
                encoder_vec)
            # pd.Print(prob_c)

            current_score, current_h, current_c, this_prob_c = self.copy_decoder(
                pre_ids_emb, 
                encoder_vec, encoder_vec_full, encoder_proj, 
                _encoder_input_ids, pre_ids,
                prob_c, att_context,
                pre_h_state, pre_c_state,
                event_embedding)

            # decoder_inputs = fluid.layers.concat(
            #     input=[att_context, pre_ids_emb], axis=1)
            # current_h, current_c = self.lstm_step(
            #         decoder_inputs, pre_h_state, pre_c_state, self.decoder_size)
            # # compute predicting probability of nomarlized word
            # current_score = fluid.layers.fc(input=current_h,
            #                       size=self.target_dict_dim,
            #                       act='softmax',
            #                       param_attr=fluid.ParamAttr(name="out_softmax_w"),
            #                       bias_attr=fluid.ParamAttr(name="out_softmax_b"))

            # # current_state = pd.fc(input=[pre_state_expanded, pre_ids_emb],
            # #                       size=decoder_size,
            # #                       act='tanh')
            # current_state_with_lod = pd.lod_reset(x=current_h, y=pre_score)
            # current_score = pd.fc(input=current_state_with_lod,
            #                       size=self.target_dict_dim,
            #                       act='softmax',
            #                       param_attr=fluid.ParamAttr(name="out_softmax_w"),
            #                       bias_attr=fluid.ParamAttr(name="out_softmax_b"))
            # print(current_score)
            topk_scores, topk_indices = pd.topk(current_score, k=self.beam_size)
            # pd.Print(topk_indices)
            # pd.Print(topk_scores)
            selected_ids, selected_scores = topk_indices, topk_scores

            # # compute accumulated score and perform beam search
            # accu_scores = pd.elementwise_add(
            #     x=pd.log(topk_scores), y=pd.reshape(pre_score, shape=[-1]), axis=0)
            # selected_ids, selected_scores = pd.beam_search(
            #     pre_ids,
            #     pre_score,
            #     topk_indices,
            #     accu_scores,
            #     self.beam_size,
            #     # end_id=self.end_id,
            #     end_id=999999,
            #     level=0)

            # pd.Print(selected_ids)
            # pd.Print(selected_scores)

            pd.increment(x=counter, value=1, in_place=True)
            # write search result and corresponding hidden layer into tensor array
            pd.array_write(current_h, array=state_h_array, i=counter)
            pd.array_write(current_c, array=state_c_array, i=counter)
            pd.array_write(selected_ids, array=ids_array, i=counter)
            pd.array_write(selected_scores, array=scores_array, i=counter)
            # pd.Print(selected_ids)
            # pd.Print(selected_scores)

            # update condition to stop loop
            length_cond = pd.less_than(x=counter, y=array_len)
            finish_cond = pd.logical_not(pd.is_empty(x=selected_ids))
            pd.logical_and(x=length_cond, y=finish_cond, out=cond)

        # pd.Print(array_len)
        # translation_ids, translation_scores = pd.beam_search_decode(
        #     ids=ids_array, scores=scores_array, beam_size=self.beam_size, end_id=self.end_id)
        # pd.Print(translation_ids)
        translation_ids, translation_ids_index = pd.tensor_array_to_tensor(ids_array, axis=1)
        translation_scores, translation_scores_index = pd.tensor_array_to_tensor(scores_array, axis=1)

        return translation_ids, translation_scores

    def bi_lstm_encoder(self, input_seq, gate_size):
        """
        # A bi-directional lstm encoder implementation.
        # Linear transformation part for input gate, output gate, forget gate
        # and cell activation vectors need be done outside of dynamic_lstm.
        # So the output size is 4 times of gate_size.
        """
        input_forward_proj = fluid.layers.fc(input=input_seq,
                                             size=gate_size * 4,
                                             act='tanh',
                                             bias_attr=False)
        forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj, size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act='tanh',
                                              bias_attr=False)
        reversed, _ = fluid.layers.dynamic_lstm(
            input=input_reversed_proj,
            size=gate_size * 4,
            is_reverse=True,
            use_peepholes=False)
        return forward, reversed

    def simple_attention(self, encoder_vec, encoder_proj, decoder_state):
        """
        # The implementation of simple attention model
        """
        decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                             size=self.decoder_size,
                                             bias_attr=False,
                                             param_attr=fluid.ParamAttr(name="att_state_proj"))
        decoder_state_expand = fluid.layers.sequence_expand(
            x=decoder_state_proj, y=encoder_proj)
        # concated lod should inherit from encoder_proj
        mixed_state = encoder_proj + decoder_state_expand
        attention_weights = fluid.layers.fc(input=mixed_state,
                                            size=1,
                                            bias_attr=False,
                                             param_attr=fluid.ParamAttr(name="att_weights"))
        attention_weights = fluid.layers.sequence_softmax(
            input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weigths_reshape, axis=0)
        # context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        context = fluid.layers.sequence_pool(input=scaled, pool_type='max')
        return context

    def lstm_step(self, x_t, hidden_t_prev, cell_t_prev, size):
        """
        lstm step
        """
        def linear(inputs, name):
            inputs = pd.concat(inputs, axis=1)
            return fluid.layers.fc(input=inputs, 
                    size=size, 
                    bias_attr=fluid.ParamAttr(name=name + "_b"),
                    param_attr=fluid.ParamAttr(name=name))
        # print("x_t", x_t)
        # print("hidden_t_prev", hidden_t_prev)
        forget_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t], "decoder_lstm_f_w"))
        input_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t], "decoder_lstm_i_w"))
        output_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t], "decoder_lstm_o_w"))
        cell_tilde = fluid.layers.tanh(x=linear([hidden_t_prev, x_t], "decoder_lstm_c_w"))

        # # pd.Print(cell_t_prev, summarize=10)
        # # pd.Print(forget_gate, summarize=10)
        # # pd.Print(input_gate, summarize=10)
        # # pd.Print(cell_tilde, summarize=10)
        # input_info = fluid.layers.elementwise_mul(
        #         x=input_gate, y=cell_tilde)
        # # print(forget_gate)
        # # print(cell_t_prev)
        # forget_info = fluid.layers.elementwise_mul(
        #         x=forget_gate, y=cell_t_prev)
        # cell_t = fluid.layers.elementwise_add(
        #         forget_gate, input_info)
        # # pd.Print(cell_t, summarize=10)
        # cell_t = fluid.layers.tanh(x=cell_t, name="decoder_lstm_step_tanh")
        # # pd.Print(output_gate, summarize=10)
        # # pd.Print(cell_t, summarize=10)
        # # hidden_t = fluid.layers.elementwise_mul(
        # #     x=output_gate, y=cell_t)
        # # hidden_t = fluid.layers.elementwise_mul(
        # #     x=output_gate, y=cell_t)
        # # pd.Print(hidden_t, summarize=10)

        cell_t = fluid.layers.sums(input=[
            fluid.layers.elementwise_mul(
                x=forget_gate, y=cell_t_prev), fluid.layers.elementwise_mul(
                    x=input_gate, y=cell_tilde)
        ])

        hidden_t = fluid.layers.elementwise_mul(
            x=output_gate, y=fluid.layers.tanh(x=cell_t))

        # print(output_gate)
        # print(cell_t)
        return hidden_t, cell_t
        # return output_gate, cell_t

    def copy_decoder(self, current_word, 
            encoder_vec, encoder_vec_full, encoder_proj, 
            _encoder_input_ids, last_ids,
            prob_c, context,
            pre_h_state, pre_c_state,
            event_embedding):
        """
        copy net decoder
        """
        # print(last_ids)
        # print(fluid.layers.reshape(last_ids, [-1, 1]).shape, _encoder_input_ids.shape)

        # last_ids_add_dim = fluid.layers.reshape(last_ids, [-1, 1], inplace=False)
        last_ids_add_dim = last_ids
        # last_ids_add_dim = fluid.layers.expand(last_ids, [1, 1])

        expand_last_ids = fluid.layers.sequence_expand_as(last_ids, _encoder_input_ids)
        expand_last_ids.stop_gradient = True
        # expand_last_ids.persistable = True
        # expand_last_ids = last_ids_add_dim
        # expand_last_ids.stop_gradient = False

        # return fluid.layers.cast(expand_last_ids, "float32")

        # print(_encoder_input_ids)
        # print(last_ids)
        # print(expand_last_ids)
        if_last_word_in_input = fluid.layers.equal(_encoder_input_ids, expand_last_ids)
        if_last_word_in_input.stop_gradient = True
        # if_last_word_in_input.persistable = True
        if_last_word_in_input = fluid.layers.cast(if_last_word_in_input, "float32")
        # print(if_last_word_in_input)
        # print(encoder_vec)
        # mask_encoder_vec = cf.split_lod_tensor(encoder_vec, if_last_word_in_input)
        # print(mask_encoder_vec)
        # weighted_encoder_vec = fluid.layers.elementwise_mul(mask_encoder_vec, prob_c, axis=0)

        # # return if_last_word_in_input
        # # return mask_encoder_vec
        # return weighted_encoder_vec

        mask_encoder_vec = fluid.layers.elementwise_mul(encoder_vec, if_last_word_in_input, axis=0)
        # print(mask_encoder_vec)
        weighted_encoder_vec = fluid.layers.elementwise_mul(mask_encoder_vec, prob_c, axis=0)
        # pd.Print(encoder_vec, summarize=10)
        # print(weighted_encoder_vec)
        # return mask_encoder_vec
        # return weighted_encoder_vec

        # selective_read = fluid.layers.sequence_pool(weighted_encoder_vec, pool_type='sum')
        selective_read = fluid.layers.sequence_pool(weighted_encoder_vec, pool_type='max')
        # pd.Print(encoder_vec, summarize=10)
        # selective_read = fluid.layers.sequence_pool(mask_encoder_vec, pool_type='sum')
        # selective_read = fluid.layers.sequence_pool(weighted_encoder_vec, pool_type='sum')
        # print(selective_read)
        # selective_read.stop_gradient = True

        # return selective_read

        decoder_inputs = fluid.layers.concat(
            input=[context, current_word, selective_read, event_embedding], axis=1)
        # print(decoder_inputs)
        # decoder_inputs = fluid.layers.concat(
        #     input=[context, current_word], axis=1)
        # current_h, current_c = pd.lstm_unit(decoder_inputs, 
        #         pre_h_state, pre_c_state, self.max_length, 
        #         self.decoder_size, 1)
        # compute predicting probability of nomarlized word

        # decoder_inputs.stop_gradient = True
        # pre_h_state.stop_gradient = True
        # pre_c_state.stop_gradient = True
        current_h, current_c = self.lstm_step(
                decoder_inputs, pre_h_state, pre_c_state, self.decoder_size)
        # pd.Print(pre_h_state)
        # pd.Print(pre_c_state)
        # pre_h_state.stop_gradient = False
        # pre_c_state.stop_gradient = False

        # current_h, current_c = pd.lstm_unit(
        #         x_t=decoder_inputs, 
        #         hidden_t_prev=pre_h_state,
        #         cell_t_prev=pre_c_state,
        #         name="decoder_lstm")
        
        # current_h = fluid.layers.fc(input=[decoder_inputs, pre_h_state], 
        #                 size=self.decoder_size, act='relu')
        # current_c = fluid.layers.fc(input=[decoder_inputs, pre_c_state], 
        #                 size=self.decoder_size, act='relu')
        
        # current_h, current_c = self.decoder_lstm(decoder_inputs, pre_h_state, pre_c_state)

        # current_h.stop_gradient = False
        # current_c.stop_gradient = False
        # print(current_h)
        # print(current_c)
        # pd.Print(current_h)
        # pd.Print(current_c)

        # outputs = fluid.layers.fc(input=current_h,
        #                       size=self.target_dict_dim,
        #                       act='softmax',
        #                       param_attr=fluid.ParamAttr(name="gen_softmax_w"),
        #                       # bias_attr=fluid.ParamAttr(name="gen_softmax_b"))
        #                       bias_attr=False)
        # return outputs, current_h, current_c, prob_c

        # pd.Print(encoder_vec, summarize=10)
        generate_score = fluid.layers.fc(input=current_h,
                              size=self.tag_dict_dim,
                              # act='softmax',
                              param_attr=fluid.ParamAttr(name="gen_softmax_w"),
                              # bias_attr=fluid.ParamAttr(name="gen_softmax_b"))
                              bias_attr=False)
        # return generate_score, current_h, current_c
        # pd.Print(encoder_vec, summarize=10)

        print(_encoder_input_ids)
        print(encoder_vec_full)
        encoder_vec_full_in = pd.reshape(
                encoder_vec_full, [-1, self.decoder_size * 2])
        copy_score_weight = fluid.layers.fc(input=encoder_vec_full_in, 
                        act='tanh',
                        size=self.decoder_size, 
                        # size=1, 
                        bias_attr=False,
                        param_attr=fluid.ParamAttr(name="copy_weight_w"))
        copy_score_weight = pd.reshape(
                copy_score_weight, [-1, self.max_length, self.decoder_size])
        # # pd.Print(copy_score_weight, summarize=10)
        # copy_score = pd.reshape(
        #         copy_score_weight, [-1, self.max_length])

        print(copy_score_weight)
        current_h_expand_seq = pd.reshape(
                current_h, [-1, 1, self.decoder_size])
        current_h_expand_seq = pd.expand(
                current_h_expand_seq, [1, self.max_length, 1])
        print(current_h_expand_seq)

        # copy_score_in = pd.concat([copy_score_weight, current_h_expand_seq], axis=2)
        # copy_score_in = pd.reshape(
        #         copy_score_in, [-1, self.decoder_size * 2])
        # copy_score = fluid.layers.fc(input=copy_score_in, 
        #             act='tanh',
        #             size=1, 
        #             bias_attr=False,
        #             param_attr=fluid.ParamAttr(name="copy_score_combine_weight_w"))
        # copy_score = pd.reshape(
        #         copy_score, [-1, self.max_length])

        copy_score_sub = pd.elementwise_mul(
                copy_score_weight, current_h_expand_seq)
                # copy_score_weight, current_h_expand_seq, axis=0)
        print(copy_score_sub)

        copy_score = pd.reduce_sum(copy_score_sub, dim=2)
        # copy_score = pd.reduce_mean(copy_score_sub, dim=2)
        print(copy_score)
        
        # copy_score_sub = pd.reshape(
        #         copy_score_sub, [-1, self.decoder_size])
        # copy_score = fluid.layers.fc(input=copy_score_sub, 
        #                 act='tanh',
        #                 size=1, 
        #                 bias_attr=False,
        #                 param_attr=fluid.ParamAttr(name="copy_score_weight_w"))
        # copy_score = pd.reshape(
        #         copy_score, [-1, self.max_length])


        # pd.Print(copy_score, summarize=10)
        # copy_score = pd.reshape(copy_score, [-1, 1])
        # print(copy_score)
        # pad_zero = pd.fill_constant(shape=[1], dtype='int64', value=0)
        # copy_score_pad = pd.sequence_pad(copy_score, pad_zero,
        #         maxlen=self.max_length, name="copy_score_padding")
        # print(copy_score_pad)
        # copy_score = copy_score_pad

        # # copy_score = pd.sequence_reshape(copy_score_pad, 
        # #         self.max_length)
        # encoder_input_mask = pd.one_hot(
        #         _encoder_input_ids, depth=self.max_length)
        # expand_copy_score = pd.elementwise_mul(
        #         encoder_input_mask, copy_score, axis=0)
        # # pd.Print(expand_copy_score, summarize=10)
        # copy_score = pd.sequence_pool(expand_copy_score,
        #         pool_type='sum')
        # # pd.Print(copy_score, summarize=10)
        # # pd.Print(copy_score, summarize=10)
        # # print(copy_score)
        # # copy_score.stop_gradient = True
        
        # # OLD Copy
        # copy_score_weight = fluid.layers.fc(input=encoder_vec, 
        #                 act='tanh',
        #                 size=self.decoder_size, 
        #                 bias_attr=False,
        #                 param_attr=fluid.ParamAttr(name="copy_weight_w"))
        # current_h_expand_seq = pd.sequence_expand_as(
        #         current_h, copy_score_weight)
        # # print(current_h_expand_seq)
        # copy_score_sub = pd.elementwise_mul(
        #         copy_score_weight, current_h_expand_seq, axis=0)
        # # print(copy_score_sub)
        # copy_score = pd.reduce_sum(copy_score_sub, dim=1)
        # # print(copy_score)
        # # pad_zero = pd.fill_constant(shape=[1], dtype='int64', value=0)
        # # copy_score_pad = pd.sequence_pad(copy_score, pad_zero,
        # #         maxlen=self.max_length, name="copy_score_padding")
        # # copy_score = pd.sequence_reshape(copy_score_pad, 
        # #         self.max_length)
        # encoder_input_mask = pd.one_hot(
        #         _encoder_input_ids, depth=self.max_length)
        # expand_copy_score = pd.elementwise_mul(
        #         encoder_input_mask, copy_score, axis=0)
        # copy_score = pd.sequence_pool(expand_copy_score,
        #         pool_type='sum')
        # # print(copy_score)
        # # copy_score.stop_gradient = True

        outputs_scores = pd.concat(
            input=[copy_score, generate_score], axis=1)
        # pd.Print(outputs_scores, summarize=10)
        outputs = pd.softmax(outputs_scores)
        # pd.Print(outputs, summarize=10)

        return outputs, current_h, current_c, prob_c
