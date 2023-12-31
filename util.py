from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random
import json


logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]


def initialize_config(config_name):
    logger.info("Running experiment: {}".format(config_name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]
    config['log_dir'] = join(config["log_root"], config_name)
    makedirs(config['log_dir'], exist_ok=True)

    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    makedirs(config['tb_dir'], exist_ok=True)

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def bucket_distance(offsets):
    """ offsets: [num spans1, num spans2] """
    # 10 semi-logscale bin: 0, 1, 2, 3, 4, (5-7)->5, (8-15)->6, (16-31)->7, (32-63)->8, (64+)->9
    logspace_distance = torch.log2(offsets.to(torch.float)).to(torch.long) + 3
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def batch_select(tensor, idx, device=torch.device('cpu')):
    """ Do selection per row (first axis). """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]

    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]

    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        selected = torch.squeeze(selected, -1)

    return selected


def count_singletons(examples):
    return sum([len([cluster for cluster in example["clusters"] if len(cluster) == 1]) for example in examples])


def discard_singletons_all(examples):
    for example in examples:
        example["clusters"] = [cluster for cluster in example["clusters"] if len(cluster) > 1]
    return examples


def discard_singletons(clusters):
    return [cluster for cluster in clusters if len(cluster) > 1]


def find_singletons_conll(file):
    singletons = {}
    with open(file) as f:
        for line in f.read().splitlines():
            if line.startswith("#") or line == "":
                continue
            clusters_str = line.split("\t")[11]
            clusters = clusters_str.split("|")
            for cluster in clusters:
                if(cluster.startswith("(")):
                    number = cluster[1:].replace(")", "")
                    if number in singletons:
                        singletons[number] = False
                    else:
                        singletons[number] = True
    return set([k for k, v in singletons.items() if v])


def filter_conll(file, output_file):
    singletons = find_singletons_conll(file)
    with open(file) as r:
        with open(output_file, "w") as w:
            for line in r.read().splitlines():
                changed = False
                if line.startswith("#") or line == "":
                    w.write(line + "\n")
                    continue
                split = line.split("\t")
                clusters_str = split[11]
                clusters = clusters_str.split("|")
                new_clusters = []
                for cluster in clusters:
                    if cluster.replace("(", "").replace(")", "") in singletons:
                        changed = True
                    else:
                        new_clusters.append(cluster)
                if changed:
                    split[11] = "|".join(new_clusters)
                    w.write("\t".join(split) + "\n")
                else:
                    w.write(line + "\n")


if __name__ == '__main__':
    # with open("./data/UD/slavic/test.czech.512.jsonlines") as f:
    #     samples = [json.loads(line) for line in f.readlines()]
    #     samples = discard_singletons(samples)
    #     with open("./data/UD/slavic/test.czech.512.filtered.jsonlines", "w") as w:
    #         json.dump(samples, w)
    filter_conll("./data/UD/test.catalan.conllu", "./data/UD/test.catalan.filtered.conllu")
