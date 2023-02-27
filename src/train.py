###############################################################################
#
# Creating Word Embeddings.
#
# This code's skeleton is the Pytorch demo:
#   https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
#
# Edited by Kelly Marchisio, Jan 2022.
###############################################################################

import argparse
from collections import Counter, OrderedDict, deque
from datetime import datetime
import iso_losses as iso
import math
import multiprocessing
import numpy as np
import os
import random
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchtext import vocab as ttvocab

TRAIN_FILE_PRINT_FREQ = 20000
NEG_SAMPLES_CHUNK = 5000
MAX_WARMUP=100000

###############################################################################
# Functions/Classes/Methods.

# sources:
# https://stackoverflow.com/questions/11040177/datetime-round-trim-number-of-digits-in-microseconds
# https://stackoverflow.com/questions/58162544/adding-timestamp-to-print-function
def ts_print(*args, **kwargs):
    t = datetime.now()
    print(t.strftime('%Y-%m-%d %H:%M:%S'), *args, **kwargs)


def bash(command):
    subprocess.run(['bash', '-c', command])


def write(vocab_itos, embeddings, outfile):
    '''Write Embeddings.

        Args:
            vocab_itos: dictionary of index:word pairs.
            embeddings: word embedding matrix.
            outfile: outfile to write embeddings.
    '''
    with open(outfile, 'w', encoding='utf-8', errors='surrogateescape') as f:
        print(len(vocab_itos), embeddings.shape[1], file=f)
        for i in range(len(vocab_itos)):
            print(vocab_itos[i],
                    ' '.join([str(j) for j in embeddings[i]]), file=f)


def make_data(args, infile, total_lines, outfile, vocab, pos_sample_rates_dict,
        neg_sample_rates_dict, shuffle=True, threads=10, count_out_len=False):
    ts_print('Making input word pairs...', flush=True)
    n = math.floor(total_lines / threads)
    processes = []
    outfile_shards = []
    infile_shards = []
    for i in range(threads):
        start = i * n
        end = start + n
        infile_shard = infile + '.' + str(i)
        infile_shards.append(infile_shard)
        bash("awk '{{if({0}<=NR && NR<{1}) print $0}}' {2} > {3}".format(start,
            end, infile, infile_shard))

        outfile_shard = outfile + '.' + str(i)
        outfile_shards.append(outfile_shard)
        p = multiprocessing.Process(target=make_data_inner, args=(args,
            infile_shard, outfile_shard, vocab, pos_sample_rates_dict,
            neg_sample_rates_dict,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
        p.close()

    bash('cat {0} > {1}'.format(' '.join(outfile_shards), outfile))
    # Move new files into a tmpdir and just delete that at the end.
    bash('rm {0} {1}'.format(' '.join(outfile_shards), ' '.join(infile_shards)))

    outfile_len = 0
    if count_out_len:
        with open(outfile, 'r') as f:
            for _ in f:
                outfile_len += 1
        ts_print('Number of input training examples:', outfile_len, flush=True)

    if shuffle:
        ts_print('Shuffling training file...', flush=True)
        # On Mac, use gshuf instead of shuf
        subprocess.run(['bash', '-c', 'gshuf -o ' + outfile + '<' + outfile])
        ts_print('Done shuffling training file.', flush=True)

    return outfile_len


def make_data_inner(args, infile, outfile, vocab, pos_sample_rates_dict,
        neg_sample_rates_dict):
    '''Make tuples with each word's context window for negative sampling.

        Note: This is a little different than how word2vec actually does it:
        they calculate the sampling frequency of keeping the word on each
        iteration:
        https://colab.research.google.com/drive/1jjV9sTmC_9oY2tDLzi0-mduLeYzWun7U
        https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c
        http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        I can simply make_data on every epoch if i want to mimic them. -
        UPDATE: I do remake data every epoch now.

    '''
    count = 0
    vocab_itos = vocab.get_itos()
    neg_sample_words, neg_sample_rates = zip(*neg_sample_rates_dict.items())

    with open(infile, 'r', encoding='utf-8', errors='surrogateescape') as inputf:
        with open(outfile, 'w', encoding='utf-8', errors='surrogateescape') as trainf:
            samples_to_add = []
            neg_samples = []
            for line in inputf:
                # Create new sentence after eliminating frequent words.
                insent = line.strip().split()
                newsent = []
                for word in insent:
                    # Undersample frequent words.
                    rand = random.random()
                    if not pos_sample_rates_dict.get(word):
                        continue
                    if pos_sample_rates_dict[word] < rand:
                        continue
                    newsent.append(word)

                for trg_pos, trg_wd in enumerate(newsent):
                    # Skip if target word not in vocab.
                    if trg_wd  not in vocab:
                        continue

                    # Add positive samples.
                    window = random.randint(1, args.window)
                    for cxt_pos in range(max(0, trg_pos - window),
                            min(len(newsent), trg_pos + window)):
                        if cxt_pos == trg_pos:
                            continue
                        cxt_wd = newsent[cxt_pos]
                        if cxt_wd  not in vocab:
                            continue
                        samples_to_add.append(cxt_wd + "\t" + trg_wd + "\t1\n")
                        count += 1

                    # Add negative samples.
                    for i in range(args.negative):
                        if not neg_samples:
                            neg_samples = random.choices(neg_sample_words,
                                    neg_sample_rates, k=NEG_SAMPLES_CHUNK)
                        neg_word = neg_samples.pop()
                        samples_to_add.append(neg_word + "\t" + trg_wd + "\t0\n")
                        count += 1

                # If write buffer is large, write it.
                if len(samples_to_add) >= TRAIN_FILE_PRINT_FREQ:
                    sample_string = ''.join(samples_to_add)
                    trainf.write(sample_string)
                    samples_to_add = []

            sample_string = ' '.join(samples_to_add)
            trainf.write(sample_string)
    return count


class SkipGramNS(nn.Module):

    def __init__(self, vocab, embedding_dim):
        super(SkipGramNS, self).__init__()
        self.vocab = vocab
        self.cxtEmbeddings = nn.Embedding(len(self.vocab), embedding_dim)
        torch.nn.init.zeros_(self.cxtEmbeddings.weight)
        self.embeddings = nn.Embedding(len(self.vocab), embedding_dim)
        # Uniform from -0.5, 0.5 divided by embedding dimension
        # Source: https://www.quora.com/How-are-vectors-initialized-in-word2vec-algorithm
        torch.nn.init.uniform_(self.embeddings.weight,
                -0.5 / embedding_dim, 0.5 / embedding_dim)


    def forward(self, context_idxs, target_idxs):
        context_vectors = self.cxtEmbeddings(context_idxs)
        target_vectors = self.embeddings(target_idxs)
        # This is the same as below from Lakhey, but makes more sense to me.
        # Maybe does more computation, though?
        #
        # https://github.com/LakheyM/word2vec/blob/master/word2vec_SGNS_git.ipynb
        # inner_products = torch.sum(torch.mul(
        #     context_vectors, target_vectors), dim=1).reshape(-1, 1)
        inner_products = torch.diagonal(
                context_vectors @ target_vectors.T).reshape(-1, 1)
        return inner_products


class SkipGram(nn.Module):

    def __init__(self, vocab, embedding_dim):
        super(SkipGram, self).__init__()
        self.vocab = vocab
        self.embeddings = nn.Embedding(len(self.vocab), embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, len(self.vocab))

    def forward(self, target_idxs):
        target_vector = self.embeddings(target_idxs)
        out = self.linear1(target_vector)
        # Note: is it OK that I used log_softmax instead of softmax?
        logprobs = F.log_softmax(out, dim=1)
        return logprobs


###############################################################################
###############################################################################

def process_dict_pairs(pair_file, max_seeds=-1):
    '''Parses a dictionary pairs file.

        Args:
            pair_file: File of dictionary pairs (seed set)
            max_seeds: Maximum # of seeds to process.  -1 == process all.

        Returns:
            Pairs as list of (srcwd, trgwd) tuples L1 and L2 vocabs as sets.

	Copied from my (Kelly's) Euc v. Graph code.
    '''
    pairs = []
    l1_words = set()
    l2_words = set()
    with open(pair_file) as f:
        for line in f:
            w1, w2 = line.split()
            w1 = w1.strip()
            w2 = w2.strip()
            if len(pairs) <= max_seeds or max_seeds == -1:
                pairs.append((w1, w2))
                l1_words.add(w1)
                l2_words.add(w2)
            else:
                return pairs, l1_words, l2_words
    return pairs, l1_words, l2_words


def word_counts(infile, vocab=None):
    total_toks = 0
    total_lines = 0
    in_vocab_total_toks = 0
    counts = Counter()
    in_vocab_counts = Counter()
    with open(infile, encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            total_lines += 1
            toks = line.strip().split()
            total_toks += len(toks)
            counts.update(toks)

            if vocab:
                in_vocab_toks = [tok for tok in toks if tok in vocab]
                in_vocab_total_toks += len(in_vocab_toks)
                in_vocab_counts.update(in_vocab_toks)

    return total_toks, counts, total_lines, in_vocab_total_toks, in_vocab_counts


def get_neg_samples(words, rates, k):
    return np.random.choice(words, (k), p=rates)


def get_neg_sample_rates_dict(word_counts):
    '''Negative sampling rates.
        Source:
            http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    '''
    sample_rates = {}
    weighted_total = sum([math.pow(i[1], 0.75) for i in word_counts.items()])
    for word in word_counts:
        weighted_rate = math.pow(word_counts[word], 0.75) / weighted_total
        sample_rates[word] = weighted_rate
    return sample_rates


def get_pos_sample_rates_dict(args, word_counts):
    '''Context word sampling rates.
        Source:
            http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    '''
    sample_rates = {}
    total = float(sum([i[1] for i in word_counts.items()]))
    for word in word_counts:
        z_w = word_counts[word] / total
        rate = min(1, (math.sqrt(z_w / args.sample) + 1) * (args.sample / z_w))
        sample_rates[word] = rate
    return sample_rates


def create_vocab(args, word_counts):
    '''Create vocab from count of word occurrences.
        Source: https://pytorch.org/text/stable/vocab.html#torchtext.vocab.Vocab
    '''
    sorted_by_freq_tuples = sorted(word_counts.items(),
            key=lambda x: x[1], reverse=True)
    ordered_word_counts= OrderedDict(sorted_by_freq_tuples)
    vocab = ttvocab.vocab(ordered_word_counts, min_freq=args.min_count)
    vocab.insert_token('<unk>', 0)
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def preprocess_to_ids(vocab, training_data):
    ts_print('Converting input word pairs to ids, excluding pairs not in vocab...',
            flush=True)
    return [[vocab[word] for word in pair] for pair in training_data
            if pair[0] in vocab and pair[1] in vocab]


def get_dataloader(args, vocab, filename):
    ts_print('Create dataloader...', flush=True)

    # Source:
    # https://medium.com/swlh/how-to-use-pytorch-dataloaders-to-work-with-enormously-large-text-files-bbd672e955a0
    class MyIterDataset(IterableDataset):

        def __init__(self, vocab, filename):
            super(MyIterDataset).__init__()
            self.vocab = vocab
            self.filename = filename

        def line_mapper(self, line):
            context_wd, target_wd, label = line.split()
            context_id = self.vocab[context_wd.strip()]
            target_id = self.vocab[target_wd.strip()]
            return context_id, target_id, np.float32(label)

        def __iter__(self):
            file_itr = open(self.filename)
            mapped_itr = map(self.line_mapper, file_itr)
            return mapped_itr

    train_data = MyIterDataset(vocab, filename)

    # I could use the built in shuffling, but I want to remake data every
    # training iteration to get different positive/negative samples.
    return DataLoader(train_data, batch_size=args.batch, shuffle=False)

#normalize, mean center, normalize for model vectors
def norm_mc_norm(model_vecs):
    normed_model_vecs = torch.nn.functional.normalize(model_vecs)
    normed_model_vecs = normed_model_vecs - torch.mean(normed_model_vecs, 0)
    return torch.nn.functional.normalize(normed_model_vecs)

def train(args, device, vocab, train_file, total_train_n, total_lines,
        pos_sample_rates_dict, neg_sample_rates_dict):
    '''
        Run training loop.

        total_train_n = total examples in training data file.
    '''
    ts_print('Setting manual seed:', args.rand_seed)
    torch.manual_seed(args.rand_seed)

    if args.negative:
        # This was helpful (source):
        # https://github.com/LakheyM/word2vec/blob/master/word2vec_SGNS_git.ipynb
        print('Running Skipgram with Negative Sampling')
        loss_function = nn.BCEWithLogitsLoss()
        model = SkipGramNS(vocab, args.size)
    else:
        print('Running naive Skipgram')
        loss_function = nn.NLLLoss()
        model = SkipGram(vocab, args.size)

    lr_scheduler = None
    if args.opt == 'SGD':
        # tHis paper uses batch size 1024 and says chainer does 1k:
        # https://link.springer.com/content/pdf/10.1007/s41019-019-0096-6.pdf
        # Another one:
        # https://west.uni-koblenz.de/assets/theses/evaluation-model-hyperparameter-choices-word2vec.pdf
        #   looks like batch size 128  LR start 2.5
        optimizer = optim.SGD(model.parameters(), args.starting_alpha)
        lambda1 = lambda batch: max(
                args.starting_alpha * 0.0001,
                1 - (batch * args.batch / (total_train_n * args.iters)))
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.starting_alpha)
        approx_batches_per_epoch = total_train_n / float(args.batch)
        approx_total_batches = approx_batches_per_epoch * args.iters
        ts_print('Approx Batches Per Epoch:',
                round(approx_batches_per_epoch, 2))
        ts_print('Approx Total Batches:', round(approx_total_batches, 2))
        if args.warmup_type == 'steps':
            warmup_steps = min(args.warmup, MAX_WARMUP)
        else:
            warmup_steps = min(math.floor(args.warmup * approx_total_batches),
                    MAX_WARMUP)
        ts_print('Warmup Steps:', warmup_steps)
        # Linear Warmup, Polynomial Decay
        lin_warm_poly_decay = (lambda batch: (batch / warmup_steps)
                if batch <= warmup_steps
                else max(0, 1 - (batch - warmup_steps) / (
                    approx_total_batches - warmup_steps)))
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                lr_lambda=lin_warm_poly_decay)

    loaded_vecs = None
    if args.mode == 'supervised':
        print('Loading static word embeddings...', flush=True)
        # note - these vecs have been normed, mean-centered, normed.
        ref_vocab, loaded_vecs = iso.load_word_vectors(args.ref_embs)
        ref_word2idx = {word: i for i, word in enumerate(ref_vocab)}
        # Get indices of src/trg seeds to be compared in loss function.
        all_seed_pairs, _, _ = process_dict_pairs(args.seeds, args.max_seeds)
        all_src_seeds = [i[0] for i in all_seed_pairs]
        all_ref_seeds = [i[1] for i in all_seed_pairs]

        src_idxs_to_compare = [vocab[src] for src, trg in
            all_seed_pairs if src in vocab and trg in ref_word2idx]
        src_used_list = [src for src, trg in
            all_seed_pairs if src in vocab and trg in ref_word2idx]
        ref_idxs = [ref_word2idx[trg] for src, trg in
            all_seed_pairs if src in vocab and trg in ref_word2idx]
        assert len(src_idxs_to_compare) == len(ref_idxs)
        print('Loaded {0} seed pairs. {1} present in training sets.'.format(
            len(all_seed_pairs), len(ref_idxs)), flush=True)
        # Extract ref vectors which are seeds.
        loaded_vecs = torch.from_numpy(loaded_vecs[ref_idxs])
        print('Loaded {0} reference vectors'.format(len(loaded_vecs)) +
                ' for isometry calculation (Some may be duplicates).')
        if args.init_embs_w_refs:
            ts_print('Initializing source space with target word embeddings')
            with torch.no_grad():
                model.embeddings.weight[src_idxs_to_compare] = loaded_vecs
    elif args.mode == 'unsupervised' and args.loss != 'skipgram':
        # If using any iso loss in unsupervised mode.
        print('Loading static word embeddings...', flush=True)
        # Load static embeddings & words in frequency order.
        # note - these vecs have been normed, mean-centered, normed.
        ref_vocab, loaded_vecs = iso.load_word_vectors(
                args.ref_embs, args.gh_n + 1)
        loaded_vecs = torch.from_numpy(loaded_vecs)
        print('Loaded {0} reference vectors'.format(len(loaded_vecs)) +
                ' for isometry calculation (Some may be duplicates).')

    # Start Training.
    ts_print('Training...', flush=True)
    last_k_losses = deque(maxlen=100)
    last_k_sg_losses = deque(maxlen=100)
    last_k_iso_losses = deque(maxlen=100)
    total_sg_loss = 0
    total_wass_loss = 0
    examples_seen = 0
    batches_seen = 0
    beta = 0
    use_iso_loss = False
    model.to(device)
    if torch.is_tensor(loaded_vecs):
        loaded_vecs = loaded_vecs.to(device)
    for iter in range(args.iters):
        ts_print('Starting iter', iter, flush=True)
        # If not the first epoch, remake/reshuffle the data. Don't worry about
        # resetting total_train_n.
        if iter > 0:
            make_data(args, args.infile, total_lines, train_file, vocab,
                    pos_sample_rates_dict, neg_sample_rates_dict,
                    shuffle=True, threads=10)
        train_dataloader = get_dataloader(args, vocab, train_file)

        for idx, (context_idxs, target_idxs, labels) in enumerate(train_dataloader):
            context_idxs = context_idxs.to(device)
            target_idxs = target_idxs.to(device)
            if (args.loss != 'skipgram' and
                    batches_seen == args.mixed_loss_start_batch):
                ts_print('Adding Isometry measure to loss function', flush=True)
                use_iso_loss = True
            model.zero_grad()

            if args.negative:
                inner_products = model.forward(context_idxs, target_idxs)
                labels = labels.reshape(-1, 1).to(device)
                sg_loss = loss_function(inner_products, labels)
            else:
                log_probs = model(target_idxs)
                sg_loss = loss_function(log_probs, context_idxs)

            last_k_sg_losses.append(sg_loss.item())

            if use_iso_loss:
                if args.mode == 'supervised': #supervised iso losses.
                    #model_vecs_tmp = model.embeddings.weight[src_idxs_to_compare]
                    model_vecs_tmp = norm_mc_norm(model.embeddings.weight[src_idxs_to_compare])
                else:
                    model_vecs_tmp = norm_mc_norm(model.embeddings.weight[:args.gh_n])
                if (args.loss == 'wass' or args.loss == 'procwass'):
                    assert args.mode == 'supervised', ('wass/procwass losses ' +
                        'must be used in supervised mode.')
                    if args.loss == 'procwass':
                        # Map the embeddings with procrustes for best fit with
                        # loaded vecs.
                        w = solve_procrustes(model_vecs_tmp, loaded_vecs)
                        model_vecs_tmp = model_vecs_tmp @ w
                    # Assuming X, Y are paired (like as seeds, in-order) compute W1 distance.
                    # Source: https://www.stat.cmu.edu/~larry/=sml/Opt.pdf p. 5, 12
                    iso_loss_unscaled = torch.linalg.norm(model_vecs_tmp - loaded_vecs)
                    iso_loss = iso_loss_unscaled / len(loaded_vecs)
                elif args.loss == 'rs':
                    iso_loss = gh.diffble_rs_distance(model_vecs_tmp,
                            loaded_vecs, device, args.mode == 'unsupervised')
                elif args.loss == 'evs':
                    iso_loss = iso.diffble_evs_distance(model_vecs_tmp,
                            loaded_vecs, device)
                last_k_iso_losses.append(iso_loss.item())
                beta = args.beta * calculate_beta_multiplier(batches_seen, args.batch,
                        total_train_n, args.iters, args.mixed_loss_start_batch,
                        mode=args.beta_mode)
                loss = (1-beta) * sg_loss + beta * iso_loss
            else:
                loss = sg_loss

            last_k_losses.append(loss.item())

            if (idx) % args.print_freq == 0:
                print('----------------')
                ts_print('Examples seen:', examples_seen)
                ts_print('Batches seen:', batches_seen)
                ts_print('Current LR:', round(
                    optimizer.state_dict()['param_groups'][0]['lr'], 8))
                if use_iso_loss:
                    ts_print('Beta =', round(beta, 6))
                    ts_print('Current SG loss', round(sg_loss.item(), 4))
                    ts_print('Current Iso loss', round(iso_loss.item(), 6))
                    ts_print('Current Weighted Batch loss', round(loss.item(), 4))
                    ts_print('Avg Weighted loss', round(
                        sum(last_k_losses) / len(last_k_losses), 4), flush=True)
                    ts_print('Avg Iso loss', round(
                        sum(last_k_iso_losses) / len(last_k_iso_losses), 6), flush=True)
                else:
                    ts_print('Current Weighted Batch loss', round(loss.item(), 4))
                ts_print('Avg SG loss', round(
                    sum(last_k_sg_losses) / len(last_k_sg_losses), 4), flush=True)

            loss.backward()
            batches_seen += 1
            examples_seen += len(context_idxs)

            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()


        ts_print('Writing embeddings for this iteration...')
        write(vocab.get_itos(), model.embeddings.weight.detach().cpu().numpy(),
                args.outdir + '/' + args.embfilename)
        model.to(device)

    ts_print('Done Training!')


def solve_procrustes(x, y):
    u, s, vt = torch.linalg.svd(y.T @ x)
    return vt.T @ u.T


def calculate_beta_multiplier(batches_seen, batch_size, total_train_n,
        total_iters, mixed_loss_start_batch, mode):
    ''' Return a beta weight based on the percentage of training completed.

        Args:
         mode: If decrease, returns starting_beta until mixed_loss_start_batch, then decreases.
           If increase, returns 0 until mixed_loss_start_batch is reached, then increases
           If constant, returns starting_beta

    '''
    approx_total_batches = (total_train_n / float(batch_size)) * total_iters
    percent_completed_after_iso_start = ((batches_seen - mixed_loss_start_batch) /
            float((approx_total_batches - mixed_loss_start_batch)))

    if mode == 'constant':
        return 1
    elif mode == 'decrease':
        if batches_seen <= mixed_loss_start_batch:
            return 1
        else:
            return 1 - percent_completed_after_iso_start
    elif mode == 'increase':
        if batches_seen <= mixed_loss_start_batch:
            return 0
        else:
            return percent_completed_after_iso_start


###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Control isomorphic...ness of word vectors.')
    parser.add_argument('infile', help='tokenized input text training file')
    parser.add_argument('outdir', help='the output directory')
    parser.add_argument('--rand-seed', type=int, default=1, help='random seed')
    parser.add_argument('--embfilename', default='embs.out',
            help='name of file for output embeddings')
    parser.add_argument('--trainfile', default='train.txt',
            help='name of file for training data')
    parser.add_argument('--lang', default='en',
            help='language (for use in Spacy tokenizer')
    parser.add_argument('--size', type=int, default=300, help='size of word vecs')
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--iters', type=int, default=10, help='training iters')
    parser.add_argument('--negative', type=int, default=10,
        help='number of negative samples to use')
    parser.add_argument('--batch', type=int, default=1000, help='batch size')
    parser.add_argument('--starting-alpha', type=float, default=0.025,
            help='starting learning rate')
    parser.add_argument('--warmup', type=float,
            help='LR warmup - # of steps, or percentage of training batches.')
    parser.add_argument('--warmup-type', choices=['steps', 'percent'],
            help='Whether --warmup is interpreted as steps ' +
            'or as a percentage of total estimated training batches.')
    parser.add_argument('--print-freq', type=int, default=1,
        help='print after how many batches')
    parser.add_argument('--min-count', type=int, default=10,
            help='vocab min-count to be included in training')
    parser.add_argument('--sample', type=float, default=0.001,
            help='source word downsampling rate')
    parser.add_argument('--vocab-max-size', type=int, default=100000,
            help='maximum size of vocabulary')
    parser.add_argument('--opt', choices=['SGD', 'Adam'], default='SGD',
            help='Which optimizer to use')
    # Wass/GH hyperparams to be added later.
    parser.add_argument('--loss',
            choices=['skipgram', 'wass', 'procwass', 'gh', 'rs', 'evs'],
            default='skipgram', help='which loss function to use.')
    parser.add_argument('--mixed-loss-start-batch', type=int, default=0,
            help='after which batch to start iso loss')
    parser.add_argument('--ref-embs', help='the reference input embeddings')
    parser.add_argument('--seeds', help='seed translations for Wass dist')
    parser.add_argument('--max-seeds', type=int, default=-1,
            help='maximum amount of seeds to use. -1 == process all.')
    parser.add_argument('--init-embs-w-refs', type=int, choices=[0, 1],
            help='If 1, initialize embeddings for the translations of seed '
            'words with the corresponding embeddings from the refrnce space.')
    parser.add_argument('--beta', type=float, default=0.5,
            help='multiplier for ratio of sg to iso loss. Default=0.5 (even'
            'weight)')
    parser.add_argument('--mode',
            choices=['unsupervised', 'supervised'], default='supervised',
            help='Supervision mode to use with iso loss.')
    parser.add_argument('--beta-mode',
            choices=['constant', 'increase', 'decrease'], default='constant',
            help='Increase, decrease, or keep beta fixed.')
    parser.add_argument('--gh-n', type=int, default=1000,
        help='How many preloaded vectors for calculating unsupervised ' +
        'gh/wass/rsim distance')
    args = parser.parse_args()
    # Validate passed arguments.
    if args.init_embs_w_refs and not (os.path.isfile(args.ref_embs) and
            os.path.isfile(args.seeds)):
            parser.error("--init-embs-w-refs requires --ref-embs and --seeds")
    if args.mode == 'supervised' and not (
        os.path.isfile(args.ref_embs) and os.path.isfile(args.seeds)):
            parser.error("iso losses in supervised mode require --ref-embs and --seeds")
    if args.mode == 'supervised' and args.loss == 'skipgram':
        parser.error('vanilla skipgram is incompatible with supervised iso '
                'iso loss mode.')
    print('Running train.py with args:', args)

    ts_print('Counting tokens in input file...', flush=True)
    total_toks, word_counter, total_lines, _, _ = word_counts(args.infile)
    ts_print('Total tokens in input file:', total_toks, flush=True)

    vocab = create_vocab(args, word_counter)
    ts_print('Vocab Size:', len(vocab), flush=True)

    # Remove words from counter if they're not in vocab.
    word_counter_copy = word_counter.copy()
    for word in word_counter:
        if word not in vocab:
            del word_counter_copy[word]
    word_counter = word_counter_copy

    # only include words if they're in vocab
    pos_sample_rates_dict = get_pos_sample_rates_dict(args, word_counter)
    neg_sample_rates_dict = get_neg_sample_rates_dict(word_counter)

    train_file = args.outdir + '/' + args.trainfile
    total_train_n = make_data(args, args.infile, total_lines, train_file, vocab,
            pos_sample_rates_dict, neg_sample_rates_dict, shuffle=True,
            threads=10, count_out_len=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch < 5:
        print('Very small batch size. Setting device to CPU')
        device="cpu"

    train(args, device, vocab, train_file, total_train_n, total_lines,
            pos_sample_rates_dict, neg_sample_rates_dict)


###############################################################################



if __name__ == '__main__':
    main()
