# preprocessing
import argparse
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fi', help="Path to the tokenized text file")
    parser.add_argument('-fo', help="Path to output file")
    parser.add_argument('-vocab', help="Path to the vocab file")
    opt = parser.parse_args()

    # opt.fi = "tok/mono.wmt16_deen.60000/valid.en"
    # # opt.fi = "tok/mono.wmt16_deen.60000/train.sub1M.en"
    # opt.fo = "./probs.txt"
    # opt.vocab = "./spm.vocab"

    lines = 0
    counter = Counter()
    with open(opt.fi) as f:
        for line in f:
            counter.update(set(line.rstrip().split(" ")))
            lines += 1

    # print(len({k: v for k, v in counter.items() if v / lines >= 0.01}))
    # print(len({k: v for k, v in counter.items() if v / lines >= 0.001}))

    with open(opt.vocab) as fv, open(opt.fo, "w") as fo:
        for line in fv:
            tok = line.rstrip().split()[0]
            count = counter.get(tok, 0)
            prob = count / lines
            fo.write(f"{tok}\t{prob}\n")
