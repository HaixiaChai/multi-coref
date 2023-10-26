import argparse
import logging
import os
import sys
import re
import collections
import json
from transformers import BertTokenizer, AutoTokenizer
import conll
import util
import udapi_io

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

trn_docname = json.load(open("./data/UD/CorefUD-1.0-public/CorefUD1.0_lang_docname_train.json","r"))
dev_docname = json.load(open("./data/UD/CorefUD-1.0-public/CorefUD1.0_lang_docname_dev.json","r"))

pos_dict = {'CCONJ': 0, 'VERB': 1, 'ADJ': 2, 'PART': 3, 'PROPN': 4, 'ADV': 5, 'NOUN': 6, 'PUNCT': 7, 'AUX': 8, 'INTJ': 9, 'SYM': 10, 'PRON': 11, 'SCONJ': 12, 'DET': 13, 'X': 14, 'NUM': 15, '_': 16, 'ADP': 17}
deprel_dict = {'cc': 0, 'list': 1, 'amod': 2, 'mark': 3, 'det': 4, 'cop': 5, 'goeswith': 6, 'parataxis': 7, 'root': 8, 'fixed': 9, 'flat': 10, 'punct': 11, 'vocative': 12, 'discourse': 13, 'csubj': 14, 'aux': 15, 'orphan': 16, 'advmod': 17, 'acl': 18, 'iobj': 19, 'conj': 20, 'xcomp': 21, 'nummod': 22, 'ccomp': 23, 'obl': 24, 'compound': 25, 'appos': 26, 'nmod': 27, 'dislocated': 28, 'case': 29, 'dep': 30, 'obj': 31, 'nsubj': 32, 'advcl': 33, 'expl': 34, '_': 35, 'reparandum': 36, '<ROOT>': 37}
word_order_dict = {'ca_ancora':0,'cs_pcedt':0,'cs_pdt':0,'de_parcorfull':1,'de_potsdamcc':1,'en_gum':0,'en_parcorfull':0,'es_ancora':0,'fr_democrat':0,'hu_szegedkoref':1,'lt_lcc':0,'pl_pcc':0,'ru_rucor':0}
dataset_dict = {'ca_ancora':0,'cs_pcedt':1,'cs_pdt':2,'de_parcorfull':3,'de_potsdamcc':4,'en_gum':5,'en_parcorfull':6,'es_ancora':7,'fr_democrat':8,'hu_szegedkoref':9,'lt_lcc':10,'pl_pcc':11,'ru_rucor':12}
lang_dict = {'ca_ancora':0,'cs_pcedt':1,'cs_pdt':1,'de_parcorfull':2,'de_potsdamcc':2,'en_gum':3,'en_parcorfull':3,'es_ancora':4,'fr_democrat':5,'hu_szegedkoref':6,'lt_lcc':7,'pl_pcc':8,'ru_rucor':9}

def skip_doc(doc_key):
    return False

def get_mention_type(upos_head):
    if upos_head=="NOUN":
        state= 0
    elif upos_head=="PROPN":
        state= 1
    elif upos_head=="PRON":
        state= 2
    elif upos_head == "_":
        state= 3
    else:
        state= 4
    return state

def get_ud_taxonomy(deprel_head):
    
    if deprel_head in ["nsubj"]:
        return 0 #core argument subject
    elif deprel_head in ["obj", "iobj"]:
        return 1 #core arguments object
    elif deprel_head in ["obl", "vocative", "expl", "dislocated"]:
        return 2 #non core dependents nominals
    elif deprel_head in ["nmod", "appos", "nummod"]:
        return 3 #nominal dependents nominals
    elif deprel_head in ["csubj","ccomp", "xcomp", "advcl", "acl"]:
        return 4 #clauses
    elif deprel_head in ["advmod", "discourse", "amod"]:
        return 5 #modifier words
    elif deprel_head in ["aux", "cop", "mark",  "det", "clf", "case"]:
        return 6 #function words
    elif deprel_head in ["conj", "cc"]:
        return 7
    elif deprel_head in ["fixed", "flat", "compound"]:
        return 8 #MWE
    elif deprel_head in ["list", "parataxis"]:
        return 9 #loose
    elif deprel_head in ["orphan", "goeswith", "reparandum"]:
        return 10 #special
    elif deprel_head in ["punct", "root", "dep", "None", '<ROOT>']:
        return 11 #other
    else:
        return 11

def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def get_sentence_map(segments, sentence_end):
    assert len(sentence_end) == sum([len(seg) - 2 for seg in segments])  # of subtokens in all segments
    sent_map = []
    sent_idx, subtok_idx = 0, 0
    for segment in segments:
        sent_map.append(sent_idx)  # [CLS]
        for i in range(len(segment) - 2):
            sent_map.append(sent_idx)
            sent_idx += int(sentence_end[subtok_idx])
            subtok_idx += 1
        sent_map.append(sent_idx)  # [SEP]
    return sent_map


class DocumentState(object):
    def __init__(self, partition, key):
        self.partition = partition
        self.doc_key = key
        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word
        self.pos = []
        self.deprel = []
        self.head_pos = []
        self.head_deprel = []
        self.mention_type = []
        self.ud_taxonomy = []
        #self.verb = []

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []
        self.segment_pos = []
        self.segment_deprel = []
        self.segment_head_pos = []
        self.segment_head_deprel = []
        self.segment_mention_type = []
        self.segment_ud_taxonomy = []
        #self.segment_verb = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = collections.defaultdict(list)

    def finalize(self):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append(subtoken_info[9])
                    if subtoken_info[4] == 'PRP':
                        self.pronouns.append(subtoken_idx)
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = util.flatten(self.segment_info)
        while first_subtoken_idx < len(subtokens_info):
            subtoken_info = subtokens_info[first_subtoken_idx]
            coref = subtoken_info[-2] if subtoken_info is not None else '-'
            if coref != '-':
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                for part in coref.split('|'):
                    if part[0] == '(':
                        if part[-1] == ')':
                            cluster_id = int(part[1:-1])
                            self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                        else:
                            cluster_id = int(part[1:])
                            self.coref_stacks[cluster_id].append(first_subtoken_idx)
                    else:
                        cluster_id = int(part[:-1])
                        start = self.coref_stacks[cluster_id].pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))
            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        #all_mentions = util.flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = util.flatten(self.segment_subtoken_map)

        # Sanity check
        # assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(util.flatten(self.segments))
        assert num_all_seg_tokens == len(util.flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)
        
        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

    def finalize_from_udapi(self, udapi_doc):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append("SPEAKER1")
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = util.flatten(self.segment_info)
        for word in udapi_doc.nodes_and_empty:
            while subtokens_info[first_subtoken_idx] is None:
                first_subtoken_idx += 1
            subtoken_info = subtokens_info[first_subtoken_idx]
            corefs = word.coref_mentions
            if word.ord != subtoken_info[0]:
                print("fd")
                pass
            if len(corefs) > 0:
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                for part in corefs:
                    if "," in part.span:
                        continue    # Skip discontinuous mentions
                    cluster_id = part.entity.eid
                    if part.span.split("-")[0] == str(word.ord):
                        if part.span.split("-")[-1] == str(word.ord):
                            self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                        else:
                            self.coref_stacks[cluster_id].append(first_subtoken_idx)
                    elif part.span.split("-")[-1] == str(word.ord):
                        start = self.coref_stacks[cluster_id].pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))
            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))
                
        merged_clusters = [list(cluster) for cluster in merged_clusters]
        #all_mentions = util.flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = util.flatten(self.segment_subtoken_map)
        pos_map = util.flatten(self.segment_pos)
        deprel_map = util.flatten(self.segment_deprel)
        head_pos_map = util.flatten(self.segment_head_pos)
        head_deprel_map = util.flatten(self.segment_head_deprel)
        mention_type_map = util.flatten(self.segment_mention_type)
        ud_taxonomy_map = util.flatten(self.segment_ud_taxonomy)
        #verb_map = util.flatten(self.segment_verb)
        
        # Sanity check
        # assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(util.flatten(self.segments))
        assert num_all_seg_tokens == len(util.flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)
        assert num_all_seg_tokens == len(pos_map)
        assert num_all_seg_tokens == len(deprel_map)
        assert num_all_seg_tokens == len(head_pos_map)
        assert num_all_seg_tokens == len(head_deprel_map)
        assert num_all_seg_tokens == len(mention_type_map)
        assert num_all_seg_tokens == len(ud_taxonomy_map)
        #assert num_all_seg_tokens == len(verb_map)
        
        # word order and language code
        file_json = dev_docname if self.partition == 'dev' else trn_docname
        for k, v in file_json.items():
            if self.doc_key in v:
                dt = dataset_dict[k]
                word_order = word_order_dict[k]
                lang = lang_dict[k]
                break
        
        return {
            "doc_key": self.doc_key,
            "dataset": dt,
            "word_order": word_order,
            "lang": lang,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "pos": pos_map,
            "deprel": deprel_map,
            "head_pos": head_pos_map,
            "head_deprel": head_deprel_map,
            "mention_typ": mention_type_map,
            "ud_taxonomy": ud_taxonomy_map,
            #"verb": head_map,
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }


def split_into_segments(document_state: DocumentState, max_seg_len, constraints1, constraints2, tokenizer):
    """ Split into segments.
        Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    """
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        # Try to split at a sentence end point
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)  # Inclusive
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            logger.info(f'{document_state.doc_key}: no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.error('Cannot split valid segment: no sentence end or token end')

        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(segment)

        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])
        document_state.segment_pos.append([document_state.pos[curr_idx]] + document_state.pos[curr_idx: end_idx + 1] + [document_state.pos[end_idx]])
        document_state.segment_deprel.append([document_state.deprel[curr_idx]] + document_state.deprel[curr_idx: end_idx + 1] + [document_state.deprel[end_idx]])
        document_state.segment_head_pos.append([document_state.head_pos[curr_idx]] + document_state.head_pos[curr_idx: end_idx + 1] + [document_state.head_pos[end_idx]])
        document_state.segment_head_deprel.append([document_state.head_deprel[curr_idx]] + document_state.head_deprel[curr_idx: end_idx + 1] + [document_state.head_deprel[end_idx]])
        document_state.segment_mention_type.append([document_state.mention_type[curr_idx]] + document_state.mention_type[curr_idx: end_idx + 1] + [document_state.mention_type[end_idx]])
        document_state.segment_ud_taxonomy.append([document_state.ud_taxonomy[curr_idx]] + document_state.ud_taxonomy[curr_idx: end_idx + 1] + [document_state.ud_taxonomy[end_idx]])
        
        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]


def get_document(partition, doc_key, language, seg_len, tokenizer, udapi_document=None):
    """ Process raw input to finalized documents """
    document_state = DocumentState(partition, doc_key)
    word_idx = -1

    # Build up documents
    last_ord = 0
    for node in udapi_document.nodes_and_empty:
        if last_ord >= node.ord:
            document_state.sentence_end[-1] = True
            # assert len(row) >= 12
        word_idx += 1
        
        word = normalize_word(node.form, language)
        #word = normalize_word(node.lemma, language)
        pos = pos_dict[node.upos]
        deprel = deprel_dict[node.deprel.split(':')[0]] if node.deprel is not None else deprel_dict[node.deps[0]['deprel'].split(':')[0]]
            
        node_parent = node.parent if node.parent is not None else node.deps[0]['parent']
        head_pos = pos_dict[node_parent.upos] if node_parent.ord != 0 else pos
        head_deprel = deprel_dict[node_parent.deprel.split(':')[0]] if node_parent.deprel is not None else deprel_dict[node.deps[0]['deprel'].split(':')[0]]
        
        mention_type = get_mention_type(node_parent.upos if node_parent.ord != 0 else node.upos)
        ud_taxonomy = get_ud_taxonomy(node_parent.deprel.split(':')[0] if node_parent.deprel is not None else deprel_dict[node.deps[0]['deprel'].split(':')[0]])
        '''
        if node.upos == 'VERB':
            verb = node.ord
        elif node_parent.upos == 'VERB':
            verb = node_parent.ord    
        else:
            try:
                while node_parent.upos != 'VERB':
                    node_parent = node_parent.parent
                verb = node_parent.ord
            except:
                verb = -1
        '''
        subtokens = tokenizer.tokenize(word)
        document_state.tokens.append(word)
        document_state.token_end += [False] * (len(subtokens) - 1) + [True]
        
        for idx, subtoken in enumerate(subtokens):
            document_state.subtokens.append(subtoken)
            info = None if idx != 0 else ([node.ord] + [node.form] + [len(subtokens)])
            #info = None if idx != 0 else ([node.ord] + [node.lemma] + [len(subtokens)])
            document_state.info.append(info)
            document_state.sentence_end.append(False)
            document_state.subtoken_map.append(word_idx)
            document_state.pos.append(pos)
            document_state.deprel.append(deprel)
            document_state.head_pos.append(head_pos)
            document_state.head_deprel.append(head_deprel)
            document_state.mention_type.append(mention_type)
            document_state.ud_taxonomy.append(ud_taxonomy)
            #document_state.verb.append(verb)
        last_ord = node.ord
    document_state.sentence_end[-1] = True

    # Split documents
    constraits1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, seg_len, constraits1, document_state.token_end, tokenizer)
    if udapi_document is not None:
        document = document_state.finalize_from_udapi(udapi_document)
    else:
        document = document_state.finalize()
    return document


def minimize_partition(partition, extension, args, tokenizer):
    #input_path = '/home/helen/Desktop/CorefUD-1.0-public/test.conllu' 
    input_path = os.path.join(args.data_dir, f'{args.language}-{partition}.{extension}')
    #output_path = '/home/helen/Desktop/CorefUD-1.0-public/test.jsonlines' 
    output_path = os.path.join(args.data_dir, f'{args.language}-{partition}.{args.max_segment_len}.jsonlines')
    doc_count = 0
    empty_count = 0
    singleton_count = 0
    output_count = 0
    logger.info(f'Minimizing {input_path}...')

    # Write documents
    with open(output_path, 'w') as output_file:
        udapi_documents = udapi_io.read_data(input_path)
        for doc in udapi_documents:
            document = get_document(partition, doc.meta["docname"], args.language, args.max_segment_len, tokenizer, udapi_documents[doc_count])
            if len(document['clusters'])== 0 and partition == 'train':
                empty_count+=1
            else:
                
                is_singleton = True
                for cluster in doc.coref_entities:
                    if len(cluster.mentions) > 1:
                        is_singleton = False
                        break
                    
                if is_singleton:
                    singleton_count+=1
                else:
                    output_file.write(json.dumps(document))
                    output_file.write('\n')
                    output_count += 1
                
            doc_count += 1 
    logger.info(f'Processed {doc_count} documents and output {output_count} documents, and {empty_count} documents without coref info are dropped. {singleton_count} singleton-documents are dropped.')

def minimize_language(args):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_name)

    # minimize_partition('dev', 'v4_gold_conll', args, tokenizer)
    # minimize_partition('test', 'v4_gold_conll', args, tokenizer)
    # minimize_partition('train', 'v4_gold_conll', args, tokenizer)

    minimize_partition('test', 'conllu', args, tokenizer)
    minimize_partition('dev', 'conllu', args, tokenizer)
    minimize_partition('train', 'conllu', args, tokenizer)

if __name__ == '__main__':
    config_name = sys.argv[1]
    config = util.initialize_config(config_name)

    os.makedirs(config.data_dir, exist_ok=True)

    minimize_language(config)
