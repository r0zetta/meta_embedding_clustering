# -*- coding: utf-8 -*-
from six.moves import cPickle
import os, sys, json, io, re, random
from string import punctuation
from itertools import combinations
from collections import Counter, deque
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
import networkx as nx
import community
import spacy
import textacy.extract
import numba

punctuation += "‘“"
nlp = spacy.load('en_core_web_sm')


# File helpers

def save_bin(item, filename):
    with open(filename, "wb") as f:
        cPickle.dump(item, f)

def load_bin(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                ret = cPickle.load(f)
        except:
            pass
    return ret

def save_json(variable, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(variable, indent=4, ensure_ascii=False))

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with io.open(filename, "r", encoding="utf-8") as f:
                ret = json.load(f)
        except:
            pass
    return ret

def save_gephi_csv(data_map, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write("Source,Target,Weight\n")
        for source, targets in data_map.items():
            if len(targets) > 0:
                for target, weight in targets.items():
                    f.write(source + "," + target + "," + str(weight) + "\n")

def write_gexf(mapping, filename, node_attrs=None, attr_names=None):
    nodes = sorted(list(set([m[0] for m in mapping]).union(set([m[1] for m in mapping]))))
    vocab = {}
    vocab_inv = {}
    for index, node in enumerate(nodes):
        label = "n" + str(index)
        vocab[node] = label
        vocab_inv[label] = node

    with open(filename, "w") as f:
        header = ""
        with open("config/gexf_header.txt", "r") as g:
            for line in g:
                header += line
        f.write(header + "\n")

        if attr_names is not None and len(attr_names) > 0:
            f.write("\t\t<attributes class=\"node\">\n")
            for index, name in enumerate(attr_names):
                f.write("\t\t\t<attribute id=\"" + str(index) + "\" title=\"" + str(name) + "\" type=\"integer\"/>\n")
            f.write("\t\t</attributes>\n")


        f.write("\t\t<nodes>\n")
        indent = '\t\t\t'
        for index, node in enumerate(nodes):
            label = vocab[node]
            entry = indent+ "<node id=\"" + str(label) + "\" label=\"" + str(node) + "\">\n"
            if attr_names is not None and len(attr_names) > 0:
                entry += indent + "\t<attvalues>\n"
                for index, name in enumerate(attr_names):
                    a = node_attrs[node]
                    entry += indent + "\t\t<attvalue for=\"" + str(index) + "\" value=\"" + str(a[index]) + "\"/>\n"
                entry += indent + "\t</attvalues>\n"
            entry += indent + "</node>\n"
            f.write(entry)
        f.write("\t\t</nodes>\n")

        f.write("\t\t<edges>\n")
        for m in mapping:
            sid = vocab[m[0]]
            tid = vocab[m[1]]
            w = m[2]
            entry = indent + "<edge source=\"" + str(sid) + "\" target=\"" + str(tid) + "\" weight=\"" + str(w) + "\"/>\n"
            f.write(entry)
        f.write("\t\t</edges>\n")
        f.write("\t</graph>\n")
        f.write("</gexf>\n")

# Preprocessing and tokenization
sw = load_json("config/stopwords.json")
stopwords = sw["en"]
stopwords.append("rt")

# Preprocess token and return it, if it is valid
def is_valid_token(r):
    hashtag = False
    if r[0] == "#":
        hashtag = True
    if r == "rt":
        return None
    if r[0] == "@":
        return None
    if r.startswith("htt"):
        return None
    if r.startswith("t.co/"):
        return None
    if "&amp;" in r:
        return None
    r = r.replace("’", "'")
    r = r.strip(punctuation)
    if r is None or len(r) < 1:
        return None
    if hashtag == True:
        if r[0] != "#":
            return "#" + r
    return r

# Tokenize tweet and return tokens.
# Returns tokens both with and without stopwords
def custom_tokenize(text):
    clean_tokens = []
    clean_tokens_with_sw = []
    raw_tokens = text.split()
    for r in raw_tokens:
        r = r.lower()
        tok = is_valid_token(r)
        if tok is not None:
            if tok not in stopwords:
                clean_tokens.append(tok)
            clean_tokens_with_sw.append(tok)
    return clean_tokens, clean_tokens_with_sw


# Functions used for clustering



num_grams = 3


@numba.jit(target='cpu', nopython=True, parallel=True)
def fast_cosine_matrix(u, M):
    scores = np.zeros(M.shape[0])
    for i in numba.prange(M.shape[0]):
        v = M[i]
        m = u.shape[0]
        udotv = 0
        u_norm = 0
        v_norm = 0
        for j in range(m):
            if (np.isnan(u[j])) or (np.isnan(v[j])):
                continue

            udotv += u[j] * v[j]
            u_norm += u[j] * u[j]
            v_norm += v[j] * v[j]

        u_norm = np.sqrt(u_norm)
        v_norm = np.sqrt(v_norm)

        if (u_norm == 0) or (v_norm == 0):
            ratio = 1.0
        else:
            ratio = udotv / (u_norm * v_norm)
        scores[i] = ratio
    return scores

def get_quick_mapping(vectors):
    add_unconnected = False
    mapping = []
    t1 = int(time.time())
    #threshold = min(0.9, (0.3 + (len(vectors)/60000)))
    threshold = 0.6
    total_calcs = ((len(vectors)*len(vectors))/2)-len(vectors)
    print("Num vectors: " + str(len(vectors)))
    step = round(len(vectors)/100)
    print("Initial threshold: " + "%.3f"%threshold)
    calcs = 0
    for index in range(len(vectors)-1):
        if index % 100 == 0:
            progress = (calcs/total_calcs)*100
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write("Progress: " + "%.2f"%progress + "%")
            sys.stdout.flush()
        scores = fast_cosine_matrix(np.array(vectors[index]), np.array(vectors[index+1:]))
        calcs += len(vectors)-(index+1)
        aw = np.argwhere(scores>=threshold)
        temp = [[index, item[0]+index+1, scores[item[0]]] for item in aw if index != item[0]]
        if len(temp) > 0:
            mapping.extend(temp)
    t2 = int(time.time())
    print()
    print("Took " + str(t2-t1) + " seconds.")
    return mapping

def get_close_neighbours(vectors, threshold):
    xsim = cosine_similarity(vectors)
    similar = []
    for x in range(len(xsim[0])):
        for y in range(x, len(xsim[1])):
            if x != y:
                if xsim[x][y] > threshold:
                    similar.append([x, y, xsim[x][y]])
    return similar

def make_text_clusters(vectors, edge_ratio=3, threshold=0):
    add_unconnected = False
    vecsindexed = vectors
    desired_edges = round(len(vecsindexed) * edge_ratio)

    trimmed_mapping = []
    if len(vectors) < 20000:
        xsim = cosine_similarity(vecsindexed)
        if threshold == 0:
            sims = []
            for x in range(len(xsim[0])-1):
                sims.extend(list(xsim[x][x+1:]))
            ind = np.argpartition(sims, desired_edges*-1)[-1]
            threshold = sims[ind]
        #print("Threshold: " + "%.4f"%threshold)

        for x in range(len(xsim[0])-1):
            row = np.array(xsim[x][x+1:])
            aw = np.argwhere(row>=threshold)
            for item in aw:
                i = item[0]
                y = i+x+1
                sim = row[i]
                trimmed_mapping.append([x, y, sim])
    else:
        mapping = get_quick_mapping(vectors)
        if threshold != 0:
            trimmed_mapping = mapping
        else:
            if len(mapping) > desired_edges:
                sims = sorted([s for x, y, s in mapping], reverse=True)
                threshold = sims[desired_edges]
                #print("Threshold: " + "%.4f"%threshold)
                trimmed_mapping = [[x, y, s] for x, y, s in mapping if s >= threshold]
            else:
                trimmed_mapping = mapping
    
    g=nx.Graph()
    g.add_weighted_edges_from(trimmed_mapping)    
    communities = community.best_partition(g)

    clusters = {}
    for node, mod in communities.items():
        if mod not in clusters:
            clusters[mod] = []
        clusters[mod].append(node)
    return clusters, trimmed_mapping

def get_sentiment(texts):
    sents = []
    for text in texts:
        blob = TextBlob(text)
        for sentence in blob.sentences:
            sents.append(sentence.sentiment.polarity)
    return np.sum(sents)

def get_cluster_relevance(texts, vectors, sns, ids):
    center = get_cluster_center(vectors)
    tweets = Counter()
    urls = Counter()
    indices = Counter()
    for index, text in enumerate(texts):
        sn = sns[index]
        id_str = ids[index]
        final = vectors[index]
        sim = cosine_similarity([center, final])[0][1]
        tweets["@" + sn + ": " + text.replace("\n", " ")] = sim
        urls["https://twitter.com/"+sn+"/status/"+id_str] = sim
        indices[index] = sim
    return indices, tweets, urls

def get_label_text(texts, vectors):
    center = get_cluster_center(vectors)
    similarities = Counter()
    for index, text in enumerate(texts):
        final = vectors[index]
        sim = cosine_similarity([center, final])[0][1]
        similarities[text.replace("\n", " ").replace("\"", "").replace("\'", "")[:20]] = sim
    most_relevant = [x for x, c in similarities.most_common(1)][0]
    return most_relevant

def get_cluster_center(vectors):
    center = np.sum(np.array(vectors), axis=0)
    return center

def get_pagerank(vectors):
    xsim = cosine_similarity(vectors)
    trimmed_mapping = []
    threshold = 0
    for x in range(len(xsim[0])-1):
        row = np.array(xsim[x][x+1:])
        aw = np.argwhere(row>=threshold)
        for item in aw:
            i = item[0]
            y = i+x+1
            sim = row[i]
            trimmed_mapping.append([x, y, sim])
    g=nx.Graph()
    g.add_weighted_edges_from(trimmed_mapping)
    scores = nx.pagerank(g)
    return scores

def rank_sentences(scores, sentences):
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    return ranked_sentences

def tokenize2(text, sw):
    clean_tokens = []
    raw_tokens = text.split()
    for r in raw_tokens:
        r = r.lower()
        tok = is_valid_token(r)
        if sw == True:
            if tok in stopwords:
                tok = None
        if tok is not None:
            clean_tokens.append(tok)
    return clean_tokens

def get_word_frequencies(texts):
    wfreq = Counter()
    for text in texts:
        toks = tokenize2(text, False)
        for t in toks:
            t = t.lower()
            wfreq[t] += 1
    return wfreq

def get_ngram_frequencies(texts, num_grams):
    prev = deque()
    gram_freq = Counter()
    for text in texts:
        toks = tokenize2(text, False)
        for t in toks:
            prev.append(t)
            if len(prev) > num_grams:
                prev.popleft()
            if len(prev) == num_grams:
                last = list(prev)
                gram = " ".join(last)
                gram_freq[gram] += 1
    return gram_freq

def get_wft(wfreq, num=5):
    wft = ""
    count = 0
    for x, c in wfreq.most_common():
        if x not in stopwords and c > 1:
            m = x + "(" + str(c) + ") "
            wft += m
            count += 1
        if count >= num:
            break
    return wft

def trim_vec_label(vec_label, threshold):
    label_counts = Counter([j for i, j in vec_label])
    valid_labels = set([l for l, c in label_counts.most_common() if c > threshold])
    trimmed_vec_label = [[i, j] for i, j in vec_label if j in valid_labels]
    return trimmed_vec_label

def get_subject_verb_object_triples(texts):
    summary_counts = Counter()
    for text in texts:
        doc = nlp(text)
        for statement in textacy.extract.subject_verb_object_triples(doc):
            subject, verb, fact = statement
            summary = "(" + str(subject) + ", " + str(verb) + ", " + str(fact) + ")"
            summary_counts[summary] += 1
    return summary_counts

def print_counter_summary(val_list):
    msg = "[ "
    valc = Counter(val_list)
    for x, c in valc.most_common(5):
        msg += x + " (" + str(c) + ") "
    msg += "]"
    return msg



