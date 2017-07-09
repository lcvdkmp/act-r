import re
import json


def read_sentences(fn):
    with open(fn, 'r') as f:
        for l in f:
            print(parse_sentence(l))


def parse_sentence(l):
    # Convert non-json to json.
    # These replacements only work because the js objects we parse are
    # really simple.
    l = l.replace("'", '"')
    l = re.sub(r"{\s*(\w)", r'{"\1', l)
    l = re.sub(r",\s*(\w)", r',"\1', l)
    l = re.sub(r"(\w):", r'\1":', l)
    l = re.sub(r",$", r'', l)
    o = json.loads(l)
    return tuple(map(str.strip, o[2]['s'].split("\n")))


read_sentences('data/fillers.txt')
