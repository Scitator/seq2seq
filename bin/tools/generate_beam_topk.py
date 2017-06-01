#! /usr/bin/env python

# based on https://github.com/google/seq2seq/blob/master/bin/tools/generate_beam_viz.py

# extracts probabilities and sequences from .npz file generated during beam search. 
# and pickles a list of the length n_samples that has beam_width most probable tuples 
# (path, logprob, prob) 
# where probs are scaled to 1. 

import numpy as np
import networkx as nx
import pickle
import tqdm
import argparse
import os


def _add_graph_level(graph, level, parent_ids, names, scores):
    """Adds a levelto the passed graph"""
    for i, parent_id in enumerate(parent_ids):
        new_node = (level, i)
        parent_node = (level - 1, parent_id)
        graph.add_node(new_node)
        graph.node[new_node]["name"] = names[i]
        graph.node[new_node]["score"] = str(scores[i])
        graph.node[new_node]["size"] = 100
        # Add an edge to the parent
        graph.add_edge(parent_node, new_node)


def create_graph(predicted_ids, parent_ids, scores, vocab=None):
    def get_node_name(pred):
        return vocab[pred] if vocab else str(pred)
    
    seq_length = predicted_ids.shape[0]
    graph = nx.DiGraph()
    for level in range(seq_length):
        names = [get_node_name(pred) for pred in predicted_ids[level]]
        _add_graph_level(graph, level + 1, parent_ids[level], names, scores[level])
    graph.node[(0, 0)]["name"] = "START"
    return graph


def get_path_to_root(graph, node):
    p = graph.predecessors(node)
    assert len(p) <= 1
    self_seq = [graph.node[node]['name'].split('\t')[0]]
    if len(p) == 0:
        return self_seq
    else:
        return self_seq + get_path_to_root(graph, p[0])
    

def main(data, vocab, top_k, separator):
    beam_data = np.load(data)
    
    with open(vocab) as file:
        vocab = file.readlines()
    vocab = [v.replace("\n", "") for v in vocab]
    vocab += ["UNK", "SEQUENCE_START", "SEQUENCE_END"]
        
    data_len = len(beam_data["predicted_ids"])
    # print(data_len)

    data_iterator = zip(beam_data["predicted_ids"],
                        beam_data["beam_parent_ids"],
                        beam_data["scores"])
    
    def _tree_node_predecessor(pos):
        return graph.node[graph.predecessors(pos)[0]]

    for row_i, (predicted_ids, parent_ids, scores) in enumerate(tqdm.tqdm(data_iterator, total=data_len)):
        graph = create_graph(
                predicted_ids=predicted_ids,
                parent_ids=parent_ids,
                scores=scores,
                vocab=vocab)
        
        pred_end_node_names = {pos for pos, d in graph.node.items()
                               if d['name'] == 'SEQUENCE_END'
                               and len(graph.predecessors(pos)) > 0
                               and _tree_node_predecessor(pos)['name'] != 'SEQUENCE_END'}

        result = [(tuple(get_path_to_root(graph, pos)[1:-1][::-1]), 
                   float(graph.node[pos]['score']))
                  for pos in pred_end_node_names]
        if len(result) == 0:
            continue
        
        filtered_result = filter(lambda x: 'SEQUENCE_END' not in x[0], result)
        s_result = sorted(filtered_result, key=lambda x: x[1], reverse=True)
        probs = np.exp(np.array(list(zip(*s_result))[1]))
#         probs = nn_probs / np.sum(nn_probs)
        result_w_prob = [(path, score, prob) for (path, score), prob in zip(s_result, probs)]
        for path, score, prob in result_w_prob[:top_k]:
            path = separator.join(path)
            print("\t".join((str(row_i), path, str(score), str(prob))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate beam search top k")
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="path to the beam search data file")
    parser.add_argument(
        "-s", "--separator", type=str, required=True,
        help="separator for data output")
    parser.add_argument(
        "-k", "--top_k", type=int, required=True,
        help="number of top k to take")
    parser.add_argument(
        "-v", "--vocab", type=str, required=False,
        help="path to the vocabulary file")
    args = parser.parse_args()
    main(args.data, args.vocab, args.top_k, args.separator)

