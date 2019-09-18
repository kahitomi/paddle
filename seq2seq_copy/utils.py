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
import pickle
import random

import paddle
import paddle.fluid as fluid
# import paddle.fluid.core as core
# import paddle.fluid.framework as framework
# from paddle.fluid.executor import Executor
# from paddle.fluid.contrib.decoder.beam_search_decoder import *

from args import *
# import model
# import data
from Constant import pos2id, entity2id, event_args, label2idx, tag2label

args = parse_args()

def get_copy_id(words_in, words, tag_dict):
    out_ids = []
    for w in words:
        # print(w, w in words_in, words)
        if w in tag_dict:
            out_ids.append(tag_dict[w])
        elif w in words_in:
            out_ids.append(words_in.index(w))
        else:
            out_ids.append(0)
    return out_ids

def data_load(data_path, 
        data=False, word2id=False, 
        entity2id=False, pos2id=False,
        word2id_output=False, event_args=False,
        generate=False):
    """
    load data from file
    """
    # data: {event_type: [(trg_position, arg_positions, sent_, entity_, pos_, tag_)], ...}
    data_ori = data.read_corpus(data_path)
    data_ori = data_ori[:10]

    data_repeat_dict = {}
    for words, entities, pos_tags, labels in data_ori:
        key = " ".join(words)
        tri = ""
        for label in labels:
            if label.endswith("-T"):
                tri = label
                break
        if key not in data_repeat_dict:
            data_repeat_dict[key] = [tri]
        else:
            data_repeat_dict[key].append(tri)

    # # Remove repeat
    # data_ori_new = []
    # for words, entities, pos_tags, labels in data_ori:
    #     key = " ".join(words)
    #     if key not in data_repeat_dict:
    #         data_ori_new.append([words, entities, pos_tags, labels])
    # data_ori = data_ori_new

    tag2label_tri = {}
    for key in tag2label:
        if key.endswith("-T"):
            tag2label_tri[key] = tag2label[key]

    print("==Load file", data_path)
    print("==Data num ", len(data_ori))
    # print(data_ori[0])
    data = []
    def reader():
        data_dict = {}
        counter = 0
        for words, entities, pos_tags, labels in data_ori:
            # print("-----")
            key = " ".join(words)

            
            # length limitation
            words = words[:args.max_length]
            entities = entities[:args.max_length]
            pos_tags = pos_tags[:args.max_length]
            labels = labels[:args.max_length]

            word_ids = [word2id.get(_, 0) for _ in words]
            entity_ids = [entity2id.get(_, 0) for _ in entities]
            pos_ids = [pos2id.get(_, 0) for _ in pos_tags]

            info_dict = {}
            info_list = []
            current_label = ""
            label_start = -1
            label_end = -1
            tri = ""
            for index, label in enumerate(labels):
                label_this = "no_label"
                if label not in ["O", "o"]:
                    items = label.split("-")
                    # label_this = items[1]
                    label_this = label
                    if label_this.startswith("B-"):
                        label_this = label_this[2:]
                    if label_this.startswith("I-"):
                        label_this = label_this[2:]
                    if label_this.endswith("-T"):
                        label_this = label_this[:-2]
                    # label_this = label.strip("B-T")
                    if label.endswith("-T"):
                        tri = label_this
                    if label_this != current_label:
                        if label_start >= 0:
                            info_dict[current_label] = [label_start, label_end]
                            info_list.append((current_label, [label_start, label_end]))
                        label_start = index
                        label_end = index
                        current_label = label_this
                    elif current_label:
                        label_end = index
                    if index == (len(labels) - 1) and \
                            current_label != "":
                        if label_start >= 0:
                            info_dict[current_label] = [label_start, label_end]
                            info_list.append((current_label, [label_start, label_end]))
                else:
                    if current_label:
                        if label_start >= 0:
                            info_dict[current_label] = [label_start, label_end]
                            info_list.append((current_label, [label_start, label_end]))
                        current_label = ""
                        label_start = -1
                        label_end = -1
            output = []
            # print(tri)
            # print(info_dict)
            # print(info_list)

            event_class = "B-" + tri + "-T"
            # print(labels)
            if event_class not in tag2label:
                continue
            event_class_id = tag2label[event_class]
            # print(event_class_id)
            
            if tri in event_args:
                output.append("B-" + tri + "-T")
                for i in range(info_dict[tri][0], \
                        (info_dict[tri][1] + 1)):
                    output.append(words[i])
                for label in event_args[tri]:
                    output.append("<S>")
                    output.append("B-" + label)
                    count = 0
                    for _label, _positions in info_list:
                        if label == _label:
                            if count > 0:
                                output.append("<S>")
                                output.append("B-" + label)
                            for i in range(_positions[0], \
                                    (_positions[1] + 1)):
                                output.append(words[i])
                            count += 1

            # output.insert(0, "<S>")
            # output.insert(0, "<S>")
            # output = words

            decoder_in = list(output)
            decoder_in.insert(0, "<S>")
            decoder_out = list(output)
            decoder_out.append("<E>")
            # decoder_in_ids = [word2id_output.get(_, 0) 
            #         for _ in decoder_in]
            # decoder_out_ids = [word2id_output.get(_, 0) 
            #         for _ in decoder_out]
            decoder_in_ids = get_copy_id(words, 
                    decoder_in, word2id_output)
            decoder_out_ids = get_copy_id(words, 
                    decoder_out, word2id_output)

            # word_index = [range(len(_)) for _ in word_ids]
            word_index = range(len(word_ids))

            # print("##", word_ids)
            # print("##", word_ids)
            item = {
                "word_ids": word_ids,
                "entity_ids": entity_ids,
                "pos_ids": pos_ids,
                "decoder_in_ids": decoder_in_ids,
                "decoder_out_ids": decoder_out_ids
            }
            # print(entities)
            # print(pos_tags)
            # print(words)
            # print(word_ids)
            # print(word_index)
            # print(labels)
            # print(decoder_out)
            # print(decoder_out_ids)
            # print(decoder_in)
            # print(decoder_in_ids)
            data.append(item)
            if not generate:
                item = [
                        word_ids, 
                        entity_ids,
                        pos_ids,
                        event_class_id,
                        word_index,
                        decoder_in_ids, 
                        decoder_out_ids
                    ]
                # print("==", item)
                event_list = []
                if key in data_repeat_dict:
                    event_list = data_repeat_dict[key]
                else:
                    event_list = [event_class]
                if key not in data_dict:
                    for i in range(args.negtive_num):
                        fake_event, fake_event_id = random.choice(list(tag2label_tri.items()))
                        if fake_event.startswith("I-"):
                            continue
                        # print(fake_event, event_class)
                        # print(event_class_id, fake_event_id)
                        if fake_event not in event_list:
                            # output = []
                            # for label in event_args[fake_event[2:-2]]:
                            #     output.append("<S>")
                            #     output.append("B-" + label)
                            # output = [fake_event] + output
                            # decoder_in = ["<S>"] + output
                            # decoder_out = output + ["<E>"]
                            # print(decoder_in)
                            decoder_in = ["<S>", fake_event]
                            decoder_out = [fake_event, "<E>"]
                            decoder_in_ids = get_copy_id(words, 
                                    decoder_in, word2id_output)
                            decoder_out_ids = get_copy_id(words, 
                                    decoder_out, word2id_output)
                            item_fake = [
                                    word_ids, 
                                    entity_ids,
                                    pos_ids,
                                    fake_event_id,
                                    word_index,
                                    decoder_in_ids, 
                                    decoder_out_ids
                                ]
                            # print(item_fake)
                            counter += 1
                            # print("+", counter)
                            #yield item_fake
                        else:
                            # print("-----")
                            pass
            else:
                item = [
                        word_ids, 
                        entity_ids,
                        pos_ids,
                        event_class_id,
                        word_index,
                        decoder_out_ids
                    ]
                event_list = []
                if key in data_repeat_dict:
                    event_list = data_repeat_dict[key]
                else:
                    event_list = [event_class]
                # if key not in data_dict:
                #     # print("==Fake")
                #     # for i in range(5):
                #     #     fake_event, fake_event_id = random.choice(list(tag2label_tri.items()))
                #     for fake_event, fake_event_id in list(tag2label_tri.items()):
                #         if fake_event.startswith("I-"):
                #             continue
                #         # print(fake_event, event_class)
                #         # print(event_class_id, fake_event_id)
                #         if fake_event not in event_list:
                #             output = []
                #             for label in event_args[fake_event[2:-2]]:
                #                 output.append("<S>")
                #                 output.append("B-" + label)
                #             output = [fake_event] + output
                #             decoder_in = ["<S>"] + output
                #             decoder_out = output + ["<E>"]
                #             # print(fake_event, event_list)
                #             # decoder_in = ["<S>", fake_event]
                #             # decoder_out = [fake_event, "<E>"]
                #             decoder_in_ids = get_copy_id(words, 
                #                     decoder_in, word2id_output)
                #             decoder_out_ids = get_copy_id(words, 
                #                     decoder_out, word2id_output)
                #             item_fake = [
                #                     word_ids, 
                #                     entity_ids,
                #                     pos_ids,
                #                     fake_event_id,
                #                     word_index,
                #                     decoder_out_ids
                #                 ]
                #             # print(item_fake)
                #             counter += 1
                #             # print("+", counter)
                #             yield item_fake
                #         else:
                #             # print("-----")
                #             pass

            # for obj in feed_list:
            #     print obj
            counter += 1
            # print("+", counter)
            yield item            
            if key not in data_dict:
                data_dict[key] = 1
    # print(data[0])
    # return data
    return reader


class generate_pr(fluid.metrics.MetricBase):
    """
    calculate event tri arg PR on generating seq
    """
    def __init__(self, name=None, word_dict={}, for_test=False):
        super(generate_pr, self).__init__(name)
        self.word_dict = word_dict
        self.correct_tri = 0 
        self.predict_tri = 0
        self.label_tri = 0
        self.correct_arg = 0
        self.predict_arg = 0
        self.label_arg = 0

        self.correct_cla = 0
        self.predict_cla = 0
        self.label_cla = 0

        self.no_tri_arg = 0

        self.for_test = for_test

        self.max_length = args.max_length

        self.end_id = -1
        for k in word_dict:
            if word_dict[k] in ["<E>"]:
                self.end_id = k
                break

        self.label_dict = []
        self.predicted = {}
        self.event_args = event_args

    def append_label(self, data_iter):
        """
        append_label
        """
        label_dict = {}
        for item in data_iter():
            word_ids = item[0]
            label_ids = item[-1]
            key = " ".join([str(_) for _ in word_ids])
            label_dict.setdefault(key, {
                    "tri_dict": {},
                    "arg_dict": {}
                })
            label_item = self._get_label_json([label_ids])
            # print("##", label_item)
            label_args, label_tri = label_item[0]
            # print(label_tri)
            for kv in label_tri:
                if kv[0] and kv[1]:
                    label_dict[key]["tri_dict"][kv[0]] = kv[1]
                    label_dict[key]["arg_dict"][kv[0]] = label_args

        # print(json.dumps(label_dict, ensure_ascii=False, indent=4))
        self.label_dict = label_dict

    def reset(self):
        """
        reset para
        """
        self.correct_tri = 0 
        self.predict_tri = 0
        self.label_tri = 0
        self.correct_arg = 0
        self.predict_arg = 0
        self.label_arg = 0

        self.no_tri_arg = 0

        self.correct_cla = 0
        self.predict_cla = 0
        self.label_cla = 0
        self.predicted = {}

        # for key in self.label_dict:
        #     self.label_tri += len(self.label_dict[key]["tri_dict"])

    def show(self):
        """
        show all count num
        """
        print("--correct_tri", self.correct_tri)
        print("--predict_tri", self.predict_tri)
        print("--label_tri", self.label_tri)
        print("--correct_arg", self.correct_arg)
        print("--predict_arg", self.predict_arg)
        print("--label_arg", self.label_arg)
        print("--correct_cla", self.correct_cla)
        print("--predict_cla", self.predict_cla)
        print("--label_cla", self.label_cla)

    def update(self, preds, labels, words_list, for_generate=False):
        """
        update para
        """
        preds = self._get_sentences(preds, labels, for_generate)
        preds_list = self._get_label_json(preds)
        labels_list = self._get_label_json(labels)
        for _predict, _words, _ori_predict, _label, _ori_label in zip(preds_list, words_list, preds, labels_list, labels):
            # print("--", _words)
            # print("  ", _ori_predict)
            # print("  ", _predict)
            preds_item, preds_tri = _predict
            labels_item, labels_tri = _label

            # print([self.word_dict[x] for x in _ori_label])
            # print(labels_tri)

            # gold_chunks = set(labels_item + labels_tri)
            # pred_chunks = set(preds_item + preds_tri)
            # self.correct_cla += len(gold_chunks & pred_chunks)
            # self.label_cla += len(gold_chunks)
            # self.predict_cla += len(pred_chunks)

            # gold_chunks = set(labels_item)
            # pred_chunks = set(preds_item)
            # self.correct_arg += len(gold_chunks & pred_chunks)
            # self.label_arg += len(gold_chunks)
            # self.predict_arg += len(pred_chunks)

            # gold_chunks = set(labels_tri)
            # pred_chunks = set(preds_tri)
            # self.correct_tri += len(gold_chunks & pred_chunks)
            # self.label_tri += len(gold_chunks)
            # self.predict_tri += len(pred_chunks)

            gold_label_tri = labels_tri[0]
            if all(gold_label_tri):
                self.label_tri += 1
            else:
                continue
            for tri_item in preds_tri:
                if gold_label_tri[0] == tri_item[0]:
                    self.predict_tri += 1
                    if gold_label_tri[1] == tri_item[1]:
                        self.correct_tri += 1


                # # print(tri)
            # for tri in tri_dict:
                # tri_mini = tri[2:-2]
                # # arguments
                # label_args = arg_dict[tri]
                # for arg_key in self.event_args[tri_mini]:
                #     arg_key = "B-" + arg_key
                #     for arg, arg_value in preds_item:
                #         if arg == arg_key and arg_value:
                #             self.predict_arg += 1
                #     for arg, arg_value in label_args:
                #         if arg == arg_key and arg_value:
                #             self.label_arg += 1
                #             for p_arg, p_arg_value in preds_item:
                #                 if p_arg == arg and \
                #                         p_arg_value == arg_value and \
                #                         tri_type_correct:
                #                     self.correct_arg += 1


            # print(tri)
            # tri = tri_item[0]
            tri = gold_label_tri[0]
            tri_mini = tri[2:-2]
            # arguments
            label_args = labels_item
            for arg_key in self.event_args[tri_mini]:
                arg_key = "B-" + arg_key
                for arg, arg_value in preds_item:
                    if arg == arg_key and arg_value:
                        self.predict_arg += 1
                for arg, arg_value in label_args:
                    if arg == arg_key and arg_value:
                        self.label_arg += 1
                        if len(preds_tri) == 0:
                            self.no_tri_arg += 1
                        for p_arg, p_arg_value in preds_item:
                            if p_arg == arg and \
                                    p_arg_value == arg_value:
                                self.correct_arg += 1

            # key = " ".join([str(_) for _ in _words])
            # if key in self.predicted:
            #     continue
            # label_item = self.label_dict[key]
            # tri_dict = label_item["tri_dict"]
            # arg_dict = label_item["arg_dict"]
            # # print("  ", tri_dict)
            # # print("  ", arg_dict)

            # # self.label_tri += len(tri_dict)
            # for tri, tri_value in preds_tri:
            #     if tri and tri_value:
            #         self.predict_tri += 1
            #         if tri in tri_dict and \
            #                 tri_dict[tri] and \
            #                 tri_value == tri_dict[tri]:
            #             print(tri, tri_value, tri_dict[tri])
            #             self.correct_tri += 1

            #             # print(tri)
            #             tri_mini = tri[2:-2]
            #             # arguments
            #             label_args = arg_dict[tri]
            #             for arg_key in self.event_args[tri_mini]:
            #                 arg_key = "B-" + arg_key
            #                 for arg, arg_value in preds_item:
            #                     if arg == arg_key and arg_value:
            #                         self.predict_arg += 1
            #                 for arg, arg_value in label_args:
            #                     if arg == arg_key and arg_value:
            #                         self.label_arg += 1
            #                         for p_arg, p_arg_value in preds_item:
            #                             if p_arg == arg and \
            #                                     p_arg_value == arg_value:
            #                                 self.correct_arg += 1

            #                 # if arg_key in label_args and \
            #                 #         label_args[arg_key]:
            #                 #     self.label_arg += 1
            #                 # if arg_key in preds_item and \
            #                 #         arg_key in label_args and \
            #                 #         preds_item[arg_key] and \
            #                 #         preds_item[arg_key] == label_args[arg_key]:
            #                 #     self.correct_arg += 1

            # self.predicted[key] = 1
        # self.show()

    def _get_sentences(self, lod, label, for_generate):
        # Split the output words by lod levels
        # print("=====Get lod pred")
        # print(lod.lod())
        # print(np.array(lod))
        if for_generate:
            end_id = self.end_id
            result = []
            for ids in np.array(lod):
                new_ids = []
                pre_id = -1
                for _id in ids:
                    if _id == end_id or \
                            _id == pre_id:
                        break
                    pre_id = _id
                    new_ids.append(_id)
                result.append(new_ids)
            return result

        if len(lod.lod()) > 0:
            lod_level_1 = lod.lod()[0]
        else:
            lod_level_1 = []
            current_lod = 0
            for sent in label:
                lod_level_1.append(current_lod)
                current_lod += len(sent)
        # print(lod_level_1)
        token_array = np.array(lod)
        result = []
        for i in six.moves.xrange(len(lod_level_1) - 1):
            # print(token_array[lod_level_1[i]:lod_level_1[i + 1]])
            sentence_list = [
                token
                for token in token_array[lod_level_1[i]:lod_level_1[i + 1]]
            ]
            # print(sentence_list)
            result.append(sentence_list)
        return result

    def _get_label_json(self, ids_list):
        item_list = []
        for ids in ids_list:
            # print(ids)
            # get words
            words = []
            for w_id in ids:
                if w_id in self.word_dict:
                    words.append(self.word_dict[w_id])
                elif w_id < self.max_length:
                    words.append(str(w_id))
                else:
                    words.append("<UNK>")
            # words = [self.word_dict.get(_, "<UNK>") for _ in ids]
            # print(words)
            key_list = []
            tri_list = []
            tri_item = []
            item = []
            words_num = len(words)
            for i, w in enumerate(words):
                if w.startswith("B-"):
                    if w.endswith("-T"):
                        tri_list.append(i)
                    key_list.append(i)
            # print(tri_list)
            # print(key_list)
            for count, i in enumerate(key_list):
                if i < (words_num - 1):
                    if count == len(key_list) - 1:
                        value = words[(i + 1):]
                    else:
                        value = words[(i + 1):(key_list[count + 1])]
                    value_strip = []
                    for w in value:
                        if w.isdigit():
                            value_strip.append(w)
                        else:
                            break
                    # value = value.strip("<SE> ")
                    value = " ".join(value_strip)
                    # if not value:
                    #     continue
                    if words[i].endswith("-T"):
                        tri_item.append((words[i], value))
                    else:
                        item.append((words[i], value))
                # else:
                #     item[words[i]] = ""

            # for k in item:
            #     if item[k] in ["<E>", "<S>"]:
            #         item[k] = ""
            #     item[k] = item[k].strip("<SE> ")
            # tri_list = [words[_] for _ in tri_list]
            # del_list = []
            # for tri_key in item:
            #     if tri_key in tri_list:
            #         tri_item[tri_key] = item[tri_key]
            #         del_list.append(tri_key)
            # for tri_key in del_list:
            #     del item[tri_key]
            # print(item, [tri_key, tri_value])
            item = list(set(item))
            tri_item = list(set(tri_item))
            item_list.append([item, tri_item])
        return item_list

    def eval(self):
        """
        get final marks
        """
        t_p = 0.0
        t_r = 0.0
        t_f1 = 0.0
        if self.predict_tri != 0:
            t_p = float(self.correct_tri) / float(self.predict_tri)
        if self.label_tri != 0:
            t_r = float(self.correct_tri) / float(self.label_tri)
        if t_p != 0 or t_r != 0:
            t_f1 = (2 * t_p * t_r) / (t_p + t_r)

        a_p = 0.0
        a_r = 0.0
        a_f1 = 0.0
        if self.predict_arg != 0:
            a_p = float(self.correct_arg) / float(self.predict_arg)
        if self.label_arg != 0:
            a_r = float(self.correct_arg) / float(self.label_arg)
        if a_p != 0 or a_r != 0:
            a_f1 = (2 * a_p * a_r) / (a_p + a_r)

        acc = 0.0
        if self.label_cla != 0:
            acc = float(self.correct_cla) / float(self.label_cla)

        return [t_p, t_r, t_f1, a_p, a_r, a_f1, acc]

    def eval_show(self):
        """
        get and show final marks
        """
        [t_p, t_r, t_f1, a_p, a_r, a_f1, acc] = self.eval()
        print("==No tri arg", 
                self.no_tri_arg)
        print("==Tri corr pred label", 
                self.correct_tri, self.predict_tri, self.label_tri)
        print("==Tri P", t_p, "R", t_r, "F1", t_f1)
        print("==Arg corr pred label", 
                self.correct_arg, self.predict_arg, self.label_arg)
        print("==Arg P", a_p, "R", a_r, "F1", a_f1)
        # print("==CLASSIFICATION ACC", acc)

        return self.correct_tri


def reduce_emb(word2id, embeddings, 
        save_prefix, all_data_list):
    id2word = {}
    for k in word2id:
        id2word[word2id[k]] = k
    words_count_dict = {}
    for data in all_data_list:
        for word_id in data[0]:
            words_count_dict.setdefault(word_id, 0)
            words_count_dict[word_id] += 1
    words_count_list = sorted(words_count_dict.items(), 
                            key=lambda x: x[1], reverse=True)
    current_value = 999999999
    count = 0
    for (key, value) in words_count_list:
        if value < current_value:
            print("--", current_value, count)
            current_value = value
        count += 1
    print("--ALL", count) 
    words_count_list = words_count_list[:8000]
    keys_keep_list = [_[0] for _ in words_count_list]
    # print(keys_keep_list)
    print(embeddings.shape)
    embeddings_mini = embeddings[keys_keep_list, :]
    print(embeddings_mini.shape)
    np.save(save_prefix + "embeddings.npy", embeddings_mini)
    words_count_list = sorted(words_count_list, 
                            key=lambda x: x[0])
    word2id_mini = {}
    for i, item in enumerate(words_count_list):
        key = id2word[item[0]]
        word2id_mini[key] = i
    print(len(word2id_mini))
    with open(save_prefix + "word2id.pkl", 'w+') as fw:
        pickle.dump(word2id_mini, fw)


if __name__ == "__main__":
    import data
    from Constant import pos2id, entity2id, event_args, label2idx, tag2label
    # word2id = data.read_dictionary("data/pre_trained_word2id.pkl")
    # embeddings = np.load("data/pre_trained_embeddings.npy")
    word2id = data.read_dictionary("data/pre_trained_mini_word2id.pkl")
    embeddings = np.load("data/pre_trained_mini_embeddings.npy")

    word2id_output = word2id.copy()
    word_ori_size = len(word2id)
    # word_mini_size = len(word2id_output)
    # word_size = word_ori_size
    # word_size = word_mini_size
    word_size = 0
    for k in tag2label:
        tag2label[k] += args.max_length
        if tag2label[k] > word_size:
            word_size = tag2label[k] 
    # word2id_output.update(tag2label)
    word2id_output = tag2label
    word2id_output["<S>"] = word_size + 1
    word2id_output["<E>"] = word_size + 2
    word_size += 3

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

    word2id_output.update(tag2label)
    data_dev = data_load("data/train_pos.txt", 
            data=data, word2id=word2id, entity2id=entity2id, 
            pos2id=pos2id, word2id_output=word2id_output,
            event_args=event_args)
    dev_batch_generator = paddle.batch(
        paddle.reader.buffered(
            data_dev, size=1000),
        batch_size=32,
        drop_last=False)
    id2word_output = {}
    for k in word2id_output:
        id2word_output[word2id_output[k]] = k

    # data_train = data_load("data/train_pos.txt", 
    #         data=data, word2id=word2id, entity2id=entity2id, 
    #         pos2id=pos2id, word2id_output=word2id_output,
    #         event_args=event_args)
    # data_dev = data_load("data/dev_pos.txt", 
    #         data=data, word2id=word2id, entity2id=entity2id, 
    #         pos2id=pos2id, word2id_output=word2id_output,
    #         event_args=event_args)
    # data_test = data_load("data/test_pos.txt", 
    #         data=data, word2id=word2id, entity2id=entity2id, 
    #         pos2id=pos2id, word2id_output=word2id_output,
    #         event_args=event_args)
    # reduce_emb(word2id, embeddings, 
    #         "data/pre_trained_copy_mini_", 
    #         [_ for _ in data_train()] + [_ for _ in data_dev()] + [_ for _ in data_test()])

    scores = generate_pr(word_dict=id2word_output)
    for batch_id, data in enumerate(dev_batch_generator()):
        # scores.update(preds=[_[-1] for _ in data], labels=[_[-1] for _ in data])
        pass
        # if batch_id > 2:
        #     break
    # scores.eval_show()
