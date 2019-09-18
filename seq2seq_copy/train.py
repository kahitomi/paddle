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

import numpy as np
import time
import os
import multiprocessing
import six
import json

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.contrib.decoder.beam_search_decoder import *

from args import *
import model
import data
from Constant import pos2id, entity2id, event_args, label2idx, tag2label
from utils import *


def train(data=data):
    # Load arguments
    args = parse_args()
    options = vars(args)
    print(json.dumps(options, ensure_ascii=False, indent=4))

    # if not conf.pre_train_word_embedding:
    #     word2id = data.read_dictionary("train_data/word2id.pkl")
    #     embeddings = data.random_embedding(word2id, conf.embedding_dim)
    # else:

    # Dictrionaries init
    word2id = data.read_dictionary("train_data/pre_trained_word2id.pkl")
    # embeddings = np.load("train_data/pre_trained_embeddings.npy")
    # word2id = data.read_dictionary("train_data/pre_trained_mini_word2id.pkl")
    # embeddings = np.load("train_data/pre_trained_mini_embeddings.npy")
    # word2id = data.read_dictionary("train_data/pre_trained_copy_mini_word2id.pkl")
    # embeddings = np.load("train_data/pre_trained_copy_mini_embeddings.npy")

    # word2id_output_mini = {}
    # for i, k in enumerate(word2id):
    #     word2id_output_mini[k] = i
    #     if i > 9100:
    #         break
    # word2id_output_mini["<S>"] = 1
    # word2id_output_mini["<E>"] = 2
    # word2id = word2id_output_mini

    word2id_output = word2id.copy()
    word_ori_size = len(word2id)
    # word_mini_size = len(word2id_output)
    # word_size = word_ori_size
    # word_size = word_mini_size

    word_size = 0
    tag_size = 0
    for k in tag2label:
        if tag2label[k] > tag_size:
            tag_size = tag2label[k] 
        tag2label[k] += args.max_length
        if tag2label[k] > word_size:
            word_size = tag2label[k] 
    # word2id_output.update(tag2label)
    word2id_output = tag2label
    word2id_output["<S>"] = word_size + 1
    word2id_output["<E>"] = word_size + 2
    word_size += 3
    tag_size += 3
    print("output size", word_size, tag_size)

    # print(type(word2id), len(word2id))
    # print(type(entity2id), len(entity2id))
    # print(type(pos2id), len(pos2id))
    # print(type(word2id_output), len(word2id_output))

    # Load data
    data_train = data_load("train_data/ace_data/train.txt", 
            data=data, word2id=word2id, entity2id=entity2id, 
            pos2id=pos2id, word2id_output=word2id_output,
            event_args=event_args)
    data_dev = data_load("train_data/ace_data/dev.txt", 
            data=data, word2id=word2id, entity2id=entity2id, 
            pos2id=pos2id, word2id_output=word2id_output,
            event_args=event_args, generate=True)
    data_test = data_load("train_data/ace_data/test.txt", 
            data=data, word2id=word2id, entity2id=entity2id, 
            pos2id=pos2id, word2id_output=word2id_output,
            event_args=event_args, generate=True)

    if args.enable_ce:
        framework.default_startup_program().random_seed = 111

    # # Training process
    # net = model.net(
    #     args.embedding_dim,
    #     args.encoder_size,
    #     args.decoder_size,
    #     word_ori_size,
    #     word_size,
    #     tag_size,
    #     False,
    #     beam_size=args.beam_size,
    #     max_length=args.max_length,
    #     source_entity_dim=len(entity2id),
    #     source_pos_dim=len(pos2id),
    #     embedding_entity_dim=args.embedding_entity_dim,
    #     embedding_pos_dim=args.embedding_pos_dim,
    #     end_id=word2id_output["<E>"])
    # avg_cost = net.avg_cost
    # feed_order = net.feeding_list
    # # Test net
    # net_test = model.net(
    #     args.embedding_dim,
    #     args.encoder_size,
    #     args.decoder_size,
    #     word_mini_size,
    #     word_size,
    #     True,
    #     beam_size=args.beam_size,
    #     max_length=args.max_length,
    #     source_entity_dim=len(entity2id),
    #     source_pos_dim=len(pos2id),
    #     embedding_entity_dim=args.embedding_entity_dim,
    #     embedding_pos_dim=args.embedding_pos_dim,
    #     end_id=word2id_output["<E>"])

    # # # clone from default main program and use it as the validation program
    # main_program = fluid.default_main_program()
    # inference_program = fluid.default_main_program().clone(for_test=True)

    # optimizer = fluid.optimizer.Adam(
    #     learning_rate=args.learning_rate,
    #     regularization=fluid.regularizer.L2DecayRegularizer(
    #         regularization_coeff=1e-5))

    # optimizer.minimize(avg_cost, no_grad_set=net.no_grad_set)
    
    # print("begin memory optimization ...")
    # # fluid.memory_optimize(train_program)
    # fluid.memory_optimize(main_program)
    # print("end memory optimization ...")


    # loss = avg_cost
    train_program = fluid.Program()
    train_startup = fluid.Program()
    # if "CE_MODE_X" in os.environ:
    #     train_program.random_seed = 110
    #     train_startup.random_seed = 110
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            # Training process
            net = model.net(
                    args.embedding_dim,
                    args.encoder_size,
                    args.decoder_size,
                    word_ori_size,
                    word_size,
                    tag_size,
                    False,
                    beam_size=args.beam_size,
                    max_length=args.max_length,
                    source_entity_dim=len(entity2id),
                    source_pos_dim=len(pos2id),
                    embedding_entity_dim=args.embedding_entity_dim,
                    embedding_pos_dim=args.embedding_pos_dim,
                    end_id=word2id_output["<E>"])
            loss = net.avg_cost
            feed_order = net.feeding_list
            # gradient clipping
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByValue(
                max=1.0, min=-1.0))

            optimizer = fluid.optimizer.Adam(
                learning_rate=args.learning_rate,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-5))
            # optimizer = fluid.optimizer.Adam(
            #     learning_rate=fluid.layers.exponential_decay(
            #         learning_rate=args.learning_rate,
            #         decay_steps=400,
            #         decay_rate=0.9,
            #         staircase=True))
            optimizer.minimize(loss)
            avg_cost = loss
            # print("begin memory optimization ...")
            # fluid.memory_optimize(train_program)
            # print("end memory optimization ...")


    test_program = fluid.Program()
    test_startup = fluid.Program()
    # if "CE_MODE_X" in os.environ:
    #     test_program.random_seed = 110
    #     test_startup.random_seed = 110
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            # Test net
            net_test = model.net(
                        args.embedding_dim,
                        args.encoder_size,
                        args.decoder_size,
                        word_ori_size,
                        word_size,
                        tag_size,
                        True,
                        beam_size=args.beam_size,
                        max_length=args.max_length,
                        source_entity_dim=len(entity2id),
                        source_pos_dim=len(pos2id),
                        embedding_entity_dim=args.embedding_entity_dim,
                        embedding_pos_dim=args.embedding_pos_dim,
                        end_id=word2id_output["<E>"])

    test_program = test_program.clone(for_test=True)
    main_program = train_program
    inference_program = test_program

    # print(type(paddle.dataset.wmt14.train(args.dict_size)))
    # print(type(paddle.reader.shuffle(
    #             data_train, buf_size=1000)))
    # print(args.enable_ce)
    # for batch_id, data in enumerate(paddle.batch(
    #         paddle.reader.shuffle(
    #             paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
    #         batch_size=args.batch_size,
    #         drop_last=False)()):
    #     print(data)
    #     break

    # Disable shuffle for Continuous Evaluation only
    if not args.enable_ce:
        train_batch_generator = paddle.batch(
            paddle.reader.shuffle(
                data_train, buf_size=1000),
            batch_size=args.batch_size,
            drop_last=False)
    else:
        train_batch_generator = paddle.batch(
            data_train,
            batch_size=args.batch_size,
            drop_last=False)
    dev_batch_generator = paddle.batch(
        paddle.reader.buffered(
            data_dev, size=1000),
        batch_size=args.batch_size,
        drop_last=False)
    test_batch_generator = paddle.batch(
        paddle.reader.buffered(
            data_test, size=1000),
        batch_size=args.batch_size,
        drop_last=False)
    # print (type(train_batch_generator))

    # Init model
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    print("device count %d" % dev_count)
    # print("theoretical memory usage: ")
    # print(fluid.contrib.memory_usage(
    #         program=main_program, batch_size=args.batch_size))

    # print("=====Init Main program")
    # exe = Executor(place)
    # # Init para
    # exe.run(framework.default_startup_program())
    # # exe = fluid.ParallelExecutor(use_cuda=args.use_gpu)
    # # os.environ['CPU_NUM'] = "2"
    # # exe = fluid.parallel_executor.ParallelExecutor(
    # #         use_cuda=args.use_gpu, num_trainers=2,
    # #         loss_name=avg_cost.name,
    # #         main_program=fluid.default_main_program())

    exe = fluid.Executor(place)
    print("=====Init train program")
    exe.run(train_startup)
    print("=====Init test program")
    exe.run(test_startup)

    # print("=====Init train exe")
    # train_exe = fluid.ParallelExecutor(
    #     use_cuda=args.use_gpu, loss_name=loss.name, main_program=train_program)

    # print("=====Init test exe")
    # test_exe = fluid.ParallelExecutor(
    #     use_cuda=args.use_gpu,
    #     main_program=test_program,
    #     share_vars_from=train_exe)

    ## Set word emb
    #print("=====Set word embedding")
    #embeddings = embeddings.astype("float32")
    #word_emb_param = fluid.global_scope().find_var(
    #    "emb").get_tensor()
    #word_emb_param.set(embeddings, place)

    print("=====Init Feeder")
    feed_list = [
        main_program.global_block().var(var_name) 
        for var_name in feed_order
    ]
    feed_list_test = [
        inference_program.global_block().var(var_name) 
        for var_name in net_test.feeding_list
    ]
    # print(feed_list)
    feeder = fluid.DataFeeder(feed_list, place)
    feeder_test = fluid.DataFeeder(feed_list_test, place)
    
    # return

    def validation(generater, test_scores):
        # Use test set as validation each pass
        test_scores.reset()
        total_loss = 0.0
        count = 0
        # val_feed_list = [
        #     inference_program.global_block().var(var_name)
        #     for var_name in net_test.feeding_list
        # ]
        # val_feeder = fluid.DataFeeder(val_feed_list, place)

        for batch_id, data in enumerate(generater()):
            # The value of batch_size may vary in the last batch
	    batch_size = len(data)

	    # Setup initial ids and scores lod tensor
	    init_ids_data = np.array([word2id_output["<S>"]
                for _ in range(batch_size)], dtype='int64')
	    init_scores_data = np.array(
		[1. for _ in range(batch_size)], dtype='float32')
	    init_ids_data = init_ids_data.reshape((batch_size, 1))
	    init_scores_data = init_scores_data.reshape((batch_size, 1))
	    init_recursive_seq_lens = [1] * batch_size
	    init_recursive_seq_lens = [
		init_recursive_seq_lens, init_recursive_seq_lens
	    ]
	    init_ids = fluid.create_lod_tensor(init_ids_data,
					       init_recursive_seq_lens, place)
	    init_scores = fluid.create_lod_tensor(init_scores_data,
						  init_recursive_seq_lens, place)

	    # Feed dict for inference
	    # feed_dict = feeder.feed([[x[0]] for x in data])
	    feed_dict = feeder_test.feed(data)
	    feed_dict['init_ids'] = init_ids
	    feed_dict['init_scores'] = init_scores

	    val_fetch_outs = exe.run(
                                inference_program,
                                # test_program(),
				feed=feed_dict,
				fetch_list=[net_test.translation_ids],
				return_numpy=False)
            # test_scores.update(
            #         preds=val_fetch_outs[0], 
            #         labels=[_[-1] for _ in data])
            # print("=====Update scores")
            test_scores.update(preds=val_fetch_outs[0], 
                    labels=[_[-1] for _ in data], 
                    words_list=[_[0] for _ in data],
                    for_generate=True)


            # val_fetch_outs = exe.run(inference_program,
            #                          feed=val_feeder.feed(data),
            #                          fetch_list=[avg_cost, net.label],
            #                          return_numpy=False)
            # test_scores.update(
            #         preds=val_fetch_outs[1], 
            #         labels=[_[-1] for _ in data], 
            #         words_list=[_[0] for _ in data])

            total_loss += 1.0
            count += 1
            # if batch_id > 0:
            #     break
        values = test_scores.eval()
        test_scores.eval_show()

        return total_loss / count, values
  
    print("=====Init scores")
    id2word_output = {}
    for k in word2id_output:
        id2word_output[word2id_output[k]] = k
    scores_train = generate_pr(word_dict=id2word_output)
    scores_train.append_label(data_train)
    scores_test = generate_pr(word_dict=id2word_output)
    scores_test.append_label(data_test)
    scores_dev = generate_pr(word_dict=id2word_output)
    scores_dev.append_label(data_dev)
    max_tri_f1 = 0.0
    max_tri_pass = -1.0
    max_arg_f1 = 0.0
    max_arg_pass = -1.0
    print("=====Start training")
    for pass_id in range(1, args.pass_num + 1):
        scores_train.reset()
        pass_start_time = time.time()
        words_seen = 0
        for batch_id, _data in enumerate(train_batch_generator()):
	    batch_size = len(_data)
            words_seen += len(_data) * 2
            # print(_data)
            # print(len(_data))
            # print(sum([len(_[0]) for _ in _data]))

	    # # Setup initial ids and scores lod tensor
	    # init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
	    # init_scores_data = np.array(
		# [1. for _ in range(batch_size)], dtype='float32')
	    # init_ids_data = init_ids_data.reshape((batch_size, 1))
	    # init_scores_data = init_scores_data.reshape((batch_size, 1))
	    # init_recursive_seq_lens = [1] * batch_size
	    # init_recursive_seq_lens = [
		# init_recursive_seq_lens, init_recursive_seq_lens
	    # ]
	    # init_ids = fluid.create_lod_tensor(init_ids_data,
					       # init_recursive_seq_lens, place)
	    # init_scores = fluid.create_lod_tensor(init_scores_data,
						  # init_recursive_seq_lens, place)

	    # # Feed dict for inference
	    # # feed_dict = feeder.feed([[x[0]] for x in _data])
	    # feed_dict = feeder.feed(_data)
	    # feed_dict['init_ids'] = init_ids
	    # feed_dict['init_scores'] = init_scores

	    # avg_cost_train, preds = exe.run(
                                # framework.default_main_program(),
                                # # test_program(),
				# feed=feed_dict,
				# fetch_list=[avg_cost, net.predict],
				# return_numpy=False)



            avg_cost_train, preds = exe.run(
                    main_program,
                    # train_program(),
                    feed=feeder.feed(_data),
                    fetch_list=[avg_cost, net.label],
                    return_numpy=False)
            # print(np.array(labels).shape)
            # print(np.array(preds).tolist())
            # print([_[-1] for _ in _data])
            #print([_[0] for _ in _data])
            avg_cost_train = np.array(avg_cost_train)
            if batch_id % 10 == 0:
                print('pass_id=%d, batch_id=%d, train_loss: %f' %
                    (pass_id, batch_id, avg_cost_train))
            
            scores_train.update(preds=preds, 
                    labels=[_[-1] for _ in _data], 
                    words_list=[_[0] for _ in _data])
            # This is for continuous evaluation only
            # if args.enable_ce and batch_id >= 100:
            # if batch_id > 0:
            #     break
        scores_train.eval_show()

        pass_end_time = time.time()
        new_max_dev = False

#         print("=====Dev test")
#         dev_loss, dev_scoress = validation(dev_batch_generator, scores_dev)
#         this_dev_tri_f1 = dev_scoress[2]
#         if this_dev_tri_f1 > max_tri_f1:
#             max_tri_f1 = this_dev_tri_f1
#             max_tri_pass = pass_id
#             new_max_dev = True
#         print("==MAX tri F1", max_tri_f1, "PASS", max_tri_pass)
#         this_dev_arg_f1 = dev_scoress[5]
#         if this_dev_arg_f1 > max_arg_f1:
#             max_arg_f1 = this_dev_arg_f1
#             max_arg_pass = pass_id
#         print("==MAX arg F1", max_arg_f1, "PASS", max_arg_pass)
#         print("=====Test test")
#         test_loss, test_scores = validation(test_batch_generator, scores_test)
#         time_consumed = pass_end_time - pass_start_time
#         words_per_sec = words_seen / time_consumed
#         # print("pass_id=%d, dev_loss: %f, test_loss: %f, words/s: %f, sec/pass: %f" %
#         #       (pass_id, dev_loss, test_loss, words_per_sec, time_consumed))
#         print("pass_id=%d, words/s: %f, sec/pass: %f" %
#               (pass_id, words_per_sec, time_consumed))

#         # This log is for continuous evaluation only
#         if args.enable_ce:
#             print("kpis\ttrain_cost\t%f" % avg_cost_train)
#             # print("kpis\tdev_cost\t%f" % dev_loss)
#             # print("kpis\ttest_cost\t%f" % test_loss)
#             print("kpis\ttrain_duration\t%f" % time_consumed)

#         if pass_id % args.save_interval == 0:
#             model_path = os.path.join(args.save_dir, str(pass_id))

#             if new_max_dev:
#                 if not os.path.isdir(model_path):
#                     os.makedirs(model_path)
#                 fluid.io.save_persistables(
#                     executor=exe,
#                     dirname=model_path,
#                     main_program=main_program)
#                 print("==Save", model_path)


if __name__ == '__main__':
    train()
