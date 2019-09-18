import pickle, random
import numpy as np
import io
from Constant import pos2id, entity2id, event_args, label2idx
import config as conf
import Constant

# def get_event2id():
#     event2id = {}
#     for k in event_args.keys().sort():


def read_corpus(corpus_path, lowcase=True):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with io.open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_,entity_, pos_, tag_ = [], [], [], []
    for line in lines:
        if line != '\n':
            [char, entity, pos, label] = line.strip().split()
            if lowcase:
                char = char.lower()
            sent_.append(char)
            entity_.append(entity)
            pos_.append(pos)
            tag_.append(label)
        else:
            data.append((sent_, entity_, pos_, tag_))
            sent_, entity_, pos_, tag_ = [], [], [], []

    # [(sent, tag)], sent: [word], tag: [tag]
    return data


def read_corpus_by_class(corpus_path, lowcase=True):
    """ read corpus and store data in a list."""

    data = {} # {event_type: [(trg_position, arg_positions, sent_, entity_, pos_, tag_)], ...}
    with io.open(corpus_path, encoding="utf-8") as fr:
        lines = fr.readlines()
    trg_position, arg_positions, sent_, entity_, pos_, tag_ = None, [], [], [], [], []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        if line != "\n":
            if (idx == 0 or lines[idx-1] == "\n") and "\t" in line:
                # trigger and arguments positions.
                positions = line.strip().split("\t")
                for idx_, position in enumerate(positions):
                    type, start, end = position.split()
                    if idx_ == 0:
                        trg_position = (type, start, end)
                    else:
                        arg_positions.append((type, start, end))
                idx += 1
                continue
            [char, entity, pos, tag] = line.strip().split()
            if lowcase:
                char = char.lower()
            sent_.append(char)
            entity_.append(entity)
            pos_.append(pos)
            tag_.append(tag)
        else:
            if trg_position[0] != "None":
                if trg_position[0] in data:
                    data[trg_position[0]].append((trg_position, arg_positions, sent_, entity_, pos_, tag_))
                else:
                    data[trg_position[0]] = [(trg_position, arg_positions, sent_, entity_, pos_, tag_)]
            trg_position, arg_positions, sent_, entity_, pos_, tag_ = None, [], [], [], [], []
        idx += 1

    return data


def vocab_build(vocab_path, corpus_path, min_count=1, lowcase=True):
    """build vocabury from specified corpus."""

    data = read_corpus(corpus_path, lowcase=lowcase)
    data_ = data.values()
    data = []
    for d_ in data_:
        data += d_
    word2id = {}
    for (trg_position, arg_positions, sent_, entity_, pos_, tag_) in data:
        for word in sent_:
            if word.isdigit():
                word = "<NUM>"
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1] # [idx, count]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [_, word_freq] in word2id.items():
        if word_freq < min_count and word != "<NUM>":
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id["<UNK>"] = new_id
    word2id["<PAD>"] = 0
    print("vocab size: {}".format(len(word2id)))
    with io.open(vocab_path, "wb") as fw:
        pickle.dump(word2id, fw)

    return word2id


def pre_trained_vocab_build(vocab_path, embedding_path, pre_trained_path, dimension = 100):
    """
    load pre-trained embeddings.
    :param vocab_path: path to save vocab.
    :param embedding_path: path to save embeddings.
    :param pre_trained_path: path to the obtained pre-trained embeddings.
    :param dimension:
    :return:
    """
    word2id = {}
    embeddings = []
    embeddings.append(np.random.uniform(-1.0, 1.0, (dimension)))
    embeddings.append(np.random.uniform(-1.0, 1.0, (dimension)))
    word2id["<PAD>"] = 0
    word2id["<UNK>"] = 1
    idx = 2
    lines = io.open(pre_trained_path, encoding="utf-8").readlines()
    for line in lines:
        segs = line.split()
        word = segs[0]
        embedding = np.array([float(itm) for itm in segs[1:]])
        word2id[word] = idx
        embeddings.append(embedding)
        idx += 1
    embeddings = np.array(embeddings)

    with io.open(vocab_path, "wb") as fw:
        pickle.dump(word2id, fw)

    np.save(embedding_path, embeddings)

def read_dictionary(vocab_path):
    with io.open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))

    return word2id

def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)

    return embedding_mat

def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def arg_position_padding(event_args, arg_positions, event, max_len = 16):
    """
    argument position in a pre-defined order, and padded to a fixed length.
    :param event_args:
    :param arg_positions:
    :param event:
    :param max_len: 16, plusing 1 on the max argument number 15, this is crucial.
    :return:
    """
    position = []
    arg_dict = {} # arg_role: (start, end).
    for arg_position in arg_positions:
        arg_dict[arg_position[0]] = (arg_position[1], arg_position[2])
    for arg in event_args[event]:
        if arg in arg_dict:
            position.append(int(arg_dict[arg][0]))
        else:
            position.append(-1)

    # to pad the position to a fixed length, the max length of all event types.
    position += [-2] * (max_len-len(position))
    return position

def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences: sequence of word id.
    :param pad_mark: index of pad for word.
    :return:
    """
    seq_len_list = [len(x) for x in sequences]
    max_len = max(seq_len_list)
    seq_list = []
    for seq in sequences:
        # seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)

    return seq_list, seq_len_list

def pad_sequences_(sequences, pad_mark=0):
    """

    :param sequences: [ [seq, ways], ...]
    :param pad_mark:
    :return:
    """
    seq_len_list = [len(x) for x in sequences]
    max_len = max(seq_len_list)
    seq_list = []
    mask = []
    for seq in sequences:
        seq_ = seq[:max_len] + [[pad_mark] * len(seq[0]) for _ in range(max(max_len - len(seq), 0))]
        seq_list.append(seq_)

        mask.append([1] * len(seq) + [0] * (max_len-len(seq)))
    return seq_list, seq_len_list, mask

def generate_label(event_args, sent, trg_position, arg_positions):
    """generabe label with respect to arguments position."""

    label = ["0"] * len(sent) # 33 for no trigger nor argument.
    trg_type, trg_start, trg_end = trg_position[0], int(trg_position[1]), int(trg_position[2])
    label[trg_start] = "1"
    label[trg_start+1: trg_end] = ["1"] * (trg_end - trg_start - 1)

    for arg_role, arg_start, arg_end in arg_positions:
        arg_start = int(arg_start)
        arg_end = int(arg_end)
        arg_label = event_args[trg_type].index(arg_role) + 2
        label[arg_start] = str(arg_label)
        label[arg_start+1: arg_end] = [str(arg_label)] * (arg_end - arg_start - 1)

    return label


def episode_yield(data, classes, word2id, negs = 160):
    """

    :param data: trg_position, arg_positions, sent_, entity_, pos_, tag_
    :param ways:
    :param shots:
    :return:
    """
    if not classes:
        classes_ = random.sample(Constant.classes, conf.ways)
        support = {}
        query = {}
        for class_ in classes_:
            samples = random.sample(data[0][class_], (conf.shots + conf.queries))
            support[class_] = samples[:conf.shots]
            query[class_] = samples[conf.shots:]
    else:
        classes_ = classes
        support = data[0]
        query = data[1]
    seqs_idx, positions_, seqs, tags, entities, poses = [], [], [], [], [], []
    seqs_query_idx, seqs_query, labels_query, labels_sent, entities_query, poses_query = [], [], [], [], [], []
    weight = []
    neg_mask = []
    for class_idx, class_ in enumerate(classes_):
        supp = support[class_]
        que = random.sample(query[class_], conf.queries)
        for sample in supp:
            trg_position = sample[0]
            arg_positions = sample[1]
            sent_ = sample[2]
            entity_ = sample[3]
            pos_ = sample[4]
            tag_ = sample[5]
            positions_.append(int(trg_position[1]))
            seqs_idx.append(sentence2id(sent_, word2id))
            seqs.append(sent_)
            tags.append(tag_)

            entities.append([entity2id[entity] for entity in entity_])
            poses.append([pos2id[pos] for pos in pos_])

        for sample in que:
            trg_position = sample[0]
            arg_positions = sample[1]
            sent_ = sample[2]
            entity_ = sample[3]
            pos_ = sample[4]
            tag_ = sample[5]
            seqs_query.append(sent_)
            seqs_query_idx.append(sentence2id(sent_, word2id))
            label = [[0] * conf.ways for _ in range(len(sent_))]
            wht = [[1] * conf.ways for _ in range(len(sent_))]
            wht[int(trg_position[1])] = [1] * conf.ways
            weight.append(wht)

            # negative sampling mask.
            nmask = [[0] * conf.ways for _ in range(len(sent_))] # seq, ways
            nmask[int(trg_position[1])] = [1] * conf.ways
            neg_indices = random.sample(range(len(sent_)), negs) if len(sent_) >= negs else range(len(sent_))
            for nidx in neg_indices:
                nmask[nidx] = [1] * conf.ways
            neg_mask.append(nmask)

            # 1 for event type matched.
            label[int(trg_position[1])][class_idx] = 1  # seq, ways.
            labels_query.append(label)

            # 1 for
            label_sent = [0] * conf.ways
            label_sent[class_idx] = 1
            labels_sent.append(label_sent)

            entities_query.append([entity2id[entity] for entity in entity_])
            poses_query.append([pos2id[pos] for pos in pos_])

    # pad sequence.
    seqs_idx, seq_len = pad_sequences(seqs_idx)
    seqs_query_idx, seq_len_query = pad_sequences(seqs_query_idx)

    entities, _ = pad_sequences(entities)
    poses, _ = pad_sequences(poses)
    entities_query, _ = pad_sequences(entities_query)
    poses_query, _ = pad_sequences(poses_query)

    # pad label.
    labels_query, _, mask = pad_sequences_(labels_query)  # ways*queries, seq, ways
    weight, _, _ = pad_sequences_(weight, pad_mark=1)
    neg_mask, _, _ = pad_sequences_(neg_mask, pad_mark=0)

    return {"support": (seqs, seqs_idx, seq_len, positions_, classes_, tags, entities, poses),
            "query": (
            seqs_query, seqs_query_idx, seq_len_query, labels_query, weight, labels_sent, entities_query, poses_query, neg_mask),
            }


if __name__ == "__main__":
    # randomly initialized embeddings.
    vocab_path = "data/word2id.pkl"
    corpus_path = "data/train.txt"
    word2id = vocab_build(vocab_path, corpus_path)
    print(word2id["<UNK>"], word2id["bush"], word2id["we"], word2id["attack"])

    # pre-trained-embeddings.
    vocab_path = "data/pre_trained_word2id.pkl"
    embedding_path = "data/pre_trained_embeddings.npy"
    pre_trained_path = "data/glove.6B.100d.txt"
    dimension = 100
    pre_trained_vocab_build(vocab_path, embedding_path, pre_trained_path, dimension)



