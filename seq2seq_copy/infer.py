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
import os
import six

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.contrib.decoder.beam_search_decoder import *

from args import *
import model
from utils import *
import data
from Constant import pos2id, entity2id, event_args, label2idx, tag2label


def infer():
     # Load arguments
    args = parse_args()
    # args.batch_size = 1

    word2id = data.read_dictionary("data/pre_trained_word2id.pkl")
    embeddings = np.load("data/pre_trained_embeddings.npy")
    # word2id = data.read_dictionary("data/pre_trained_copy_mini_word2id.pkl")
    # embeddings = np.load("data/pre_trained_copy_mini_embeddings.npy")

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

    # # Dictrionaries init
    # word2id = data.read_dictionary("data/pre_trained_word2id.pkl")
    # embeddings = np.load("data/pre_trained_embeddings.npy")
    # word2id_output = word2id.copy()
    # word_mini_size = len(word2id)
    # word_size = 0
    # for k in tag2label:
    #     tag2label[k] += word_mini_size
    #     if tag2label[k] > word_size:
    #         word_size = tag2label[k] 
    # tag2label["<S>"] = word_size + 1
    # tag2label["<E>"] = word_size + 2
    # word_size += 3
    # word2id_output.update(tag2label)
    # # print(type(word2id), len(word2id))
    # # print(type(entity2id), len(entity2id))
    # # print(type(pos2id), len(pos2id))
    # # print(type(word2id_output), len(word2id_output))
    id2entity = {}
    for k in entity2id:
        id2entity[entity2id[k]] = k
    id2word = {}
    for k in word2id:
        id2word[word2id[k]] = k
    id2word_output = {}
    for k in word2id_output:
        id2word_output[word2id_output[k]] = k
    src_dict, trg_dict = id2word, id2word_output

    # Load data
    # data_train = data_load("data/train_pos.txt", 
    #         data=data, word2id=word2id, entity2id=entity2id, 
    #         pos2id=pos2id, word2id_output=word2id_output,
    #         event_args=event_args)
    data_train = data_load("data/ace_data/train.txt", 
            data=data, word2id=word2id, entity2id=entity2id, 
            pos2id=pos2id, word2id_output=word2id_output,
            event_args=event_args, generate=True)
    data_dev = data_load("data/ace_data/dev.txt", 
            data=data, word2id=word2id, entity2id=entity2id, 
            pos2id=pos2id, word2id_output=word2id_output,
            event_args=event_args, generate=True)
    data_test = data_load("data/ace_data/test.txt", 
            data=data, word2id=word2id, entity2id=entity2id, 
            pos2id=pos2id, word2id_output=word2id_output,
            event_args=event_args, generate=True)
    # data_test = data_train

    print("=====Init scores")
    scores = generate_pr(word_dict=id2word_output)
    scores.append_label(data_test)

    # Inference
    net = model.net(
        args.embedding_dim,
        args.encoder_size,
        args.decoder_size,
        word_ori_size,
        word_size,
        tag_size,
        True,
        # False,
        beam_size=args.beam_size,
        max_length=args.max_length,
        source_entity_dim=len(entity2id),
        source_pos_dim=len(pos2id),
        embedding_entity_dim=args.embedding_entity_dim,
        embedding_pos_dim=args.embedding_pos_dim,
        end_id=word2id_output["<E>"])

    # test_batch_generator = paddle.batch(
    #     paddle.reader.shuffle(
    #         paddle.dataset.wmt14.test(args.dict_size), buf_size=1000),
    #     batch_size=args.batch_size,
    #     drop_last=False)

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

    print("begin memory optimization ...")
    # fluid.memory_optimize(train_program)
    fluid.memory_optimize(framework.default_main_program())
    print("end memory optimization ...")

    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())
    # # exe = fluid.ParallelExecutor(use_cuda=args.use_gpu)
    # os.environ['CPU_NUM'] = "2"
    # exe = fluid.parallel_executor.ParallelExecutor(
    #         use_cuda=args.use_gpu, num_trainers=2,
    #         # loss_name=avg_cost.name,
    #         main_program=fluid.default_main_program())

    # LOAD Model
    model_path = os.path.join(args.save_dir, str(args.load_pass_num))
    fluid.io.load_persistables(
        executor=exe,
        dirname=model_path,
        main_program=framework.default_main_program())
    print("==Model loaded", args.save_dir)

    translation_ids = net.translation_ids
    translation_scores = net.translation_scores
    feed_order = net.feeding_list

    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    # print(feed_list)
    feeder = fluid.DataFeeder(feed_list, place)
    scores.reset()
    for batch_id, _data in enumerate(test_batch_generator()):
        print("=====", batch_id, len(_data))
        # The value of batch_size may vary in the last batch
        batch_size = len(_data)

        # Setup initial ids and scores lod tensor
        # init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
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
        # print(init_ids_data.shape)
        # print(init_recursive_seq_lens)
        # print(init_ids.lod())
        # print(init_scores.lod())

        # Feed dict for inference
        feed_dict = feeder.feed([x for x in _data])
        feed_dict['init_ids'] = init_ids
        feed_dict['init_scores'] = init_scores

        print("=====")
        fetch_outs = exe.run(framework.default_main_program(),
                             feed=feed_dict,
                             fetch_list=[translation_ids, translation_scores],
                             # fetch_list=[translation_ids],
                             return_numpy=False)
        # print(np.array(fetch_outs[0]))
        # print(np.array(fetch_outs[0]).shape)
        print("=====Update scores")
        scores.update(preds=fetch_outs[0], 
                labels=[_[-1] for _ in _data], 
                words_list=[_[0] for _ in _data],
                for_generate=True)
        # Split the output words by lod levels
        end_id = word2id_output["<E>"]
        result = []
        paragraphs = []
        for ids in np.array(fetch_outs[0]):
            # print("##", ids.shape)
            # print("##", ids)
            new_ids = []
            new_words = []
            pre_id = -1
            for _id in ids:
                if _id == end_id or \
                        _id == pre_id:
                    break
                pre_id = _id
                new_ids.append(_id)
                if _id < args.max_length:
                    new_words.append(str(_id))
                else:
                    new_words.append(trg_dict[_id])
            result.append(new_ids)
            paragraphs.append(new_words)

        # lod_level_1 = fetch_outs[0].lod()[1]
        # token_array = np.array(fetch_outs[0])
        # result = []
        # for i in six.moves.xrange(len(lod_level_1) - 1):
        #     sentence_list = [
        #         trg_dict[token]
        #         for token in token_array[lod_level_1[i]:lod_level_1[i + 1]]
        #     ]
        #     sentence = " ".join(sentence_list[1:-1])
        #     result.append(sentence)
        # lod_level_0 = fetch_outs[0].lod()[0]
        # paragraphs = [
        #     result[lod_level_0[i]:lod_level_0[i + 1]]
        #     for i in six.moves.xrange(len(lod_level_0) - 1)
        # ]

        # target_sentence_list = [" ".join(
        #         [trg_dict[__] 
        #         for __ in _[-1]]) 
        #         for _ in _data]
        target_sentence_list = []
        for item in _data:
            target_words = []
            for _id in item[-1]:
                if _id < args.max_length:
                    target_words.append(str(_id))
                else:
                    target_words.append(trg_dict[_id])
            target_sentence_list.append(
                    " ".join(target_words))
        source_sentence_list = []
        source_entity_list = []
        for item in _data:
            target_words = []
            for _id in item[0]:
                target_words.append(src_dict[_id])
            source_sentence_list.append(target_words)
            entity_tag = []
            for _id in item[1]:
                entity_tag.append(id2entity[_id])
            source_entity_list.append(entity_tag)


        print("=====Print text")
        for paragraph, sentence, source , entities in \
                zip(paragraphs, target_sentence_list, \
                source_sentence_list, source_entity_list):
            print("-----")
            new_words = []
            indexes = range(len(source))
            for i, word, entity in zip(indexes, source, entities):
                new_words.append(word + "(" + str(i) + " " + entity + ")")
            print(" ".join(new_words))
            print("=Predict:", " ".join(paragraph[1:]))
            print("=Label:", sentence)

    scores.eval_show()


if __name__ == '__main__':
    infer()
