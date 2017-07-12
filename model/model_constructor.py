import re
import json
import csv
from model import Model

# TODO: entries -> entries


def format_word(w):
    return w.strip().replace('.', '').lower()


# TODO: support for model parameters
class ModelConstructor():
    def __init__(self, sentence_filepath,
                 word_freq_csv, advanced=True, **kwargs):
        self.kwargs = kwargs
        self.entries = []
        self.advanced = advanced
        self.read_sentences(sentence_filepath, self.advanced)

        print("entries successfully parsed: {}/{}"
              .format(len(self.entries), self.num_total_entries))

        self.parse_freq_csv(word_freq_csv)
        print("All frequency information found")
        self.entries = [self.entries[0]]

    def read_sentences(self, fn, vc):
        # TODO: also retrieve second sentence
        def rem_dot(wl):
            if wl[-1][-1] == ".":
                wl[-1] = wl[-1][:-1]
            return wl

        def parse_sentence(l, i):
            ss, nl = lex_sentence(l)
            wl = [ss[0].split(" "), ss[1].split(" ")]
            wl = [rem_dot(x) for x in wl]
            wl = [[format_word(w) for w in x] for x in wl]
            wl = [list(filter(None, l)) for l in wl]
            nl = list(map(str.strip, nl))
            if not vc:
                return wl, nl

            sl = wl
            wl = sl[0]
            il = []
            # TODO: also check second sentence
            for n in nl:
                try:
                    ix = wl.index(n)
                except ValueError:
                    raise Warning(('"{}": Line {}: Noun "{}" not in'
                                   ' sentence. Skipping sentence...'
                                   ).format(fn, i, n))
                il += [ix]
            if len(il) > 2:
                raise Warning('Line {} of file "{}" contains first'
                              ' sentence with more than two nouns.'
                              ' Skipping sentence...')
            if 1 not in il:
                raise Warning(('"{}": Line {}: Second word "{}" not a'
                               ' noun!. Skipping sentence...'
                               ).format(fn, i, wl[1]))
            i2 = [i for i in il if i != 1][0]
            indicator = wl[i2 - 1]
            if wl.count(indicator) != 1:
                raise Warning(('"{}": Line {}: Indicator for second noun not'
                               ' unique! Indicator: "{}"'
                               '. Skipping sentence...'
                               ).format(fn, i, indicator))
            return sl, nl, indicator

        self.num_total_entries = 0
        with open(fn, 'r') as f:
            for i, l in enumerate(f):
                self.num_total_entries += 1
                try:
                    self.entries += [parse_sentence(l, i + 1)]
                except Warning as w:
                    print("Warning:", w)
                    continue

    def parse_freq_csv(self, fn):
        with open(fn, 'r') as f:
            sh = 0
            eh = 0
            nh = 0
            reader = csv.DictReader(f, delimiter=',')
            for r in reader:
                if self.entries[eh][0][nh][sh] != format_word(r['Word']):
                    print("skipping {}".format(format_word(r['Word'])))
                    continue

                # Miliseconds to seconds
                self.entries[eh][0][nh][sh] = (self.entries[eh][0][nh][sh],
                                               float(r['RT'] / 1000))
                sh += 1
                if sh == len(self.entries[eh][0][nh]):
                    if nh == 0:
                        nh += 1
                        sh = 0
                    else:
                        nh = 0
                        sh = 0
                        eh += 1
            if (eh, nh, sh) != (len(self.entries), 0, 0):
                raise Warning("Not all entries have corresponding"
                              "frequencies!")

            # self.freqs = {format_word(r['Word']): r['RT'] for r in reader}

    def model_generator(self, **kwargs):
        if not self.advanced:
            for wl, nl in self.entries:
                wl = [[w for w, _ in l] for l in wl]
                lex = set([w for l in wl for w in l])
                yield Model([wl], lex, advanced=self.advanced,
                            **self.kwargs, model_params=kwargs)
        # TODO: else

    def freqs(self):
        for e in self.entries:
            for l in e[0]:
                for _, f in l:
                    yield f

def lex_sentence(l):
    # Convert non-json to json.
    # These replacements only work because the js objects we parse are
    # really simple.
    l = l.replace("'", '"')
    l = re.sub(r"{\s*(\w)", r'{"\1', l)
    l = re.sub(r",\s*(\w*):", r',"\1:', l)
    l = re.sub(r"(\w):", r'\1":', l)
    l = re.sub(r",$", r'', l)
    o = json.loads(l)

    try:
        s_i = o.index('DashedSentence') + 1
        nl_i = o.index('Question') + 1
    except ValueError:
        raise Exception('The input file is of an incorrect'
                        ' shape!\nLine: "{}"'.format(l))

    # XXX: Here we replace " by '. We assume any occurance of " has been caused
    # by the regex replace used to convert javascript to json.
    s = tuple(map(str.strip, o[s_i]['s'].replace('"', "'").split("\n")))
    nl = o[nl_i]['as']
    return (s, nl)


# mc = ModelConstructor("data/fillers.txt",
#                       "data/results_fillers_RTs.csv", advanced=False)
# mc2 = ModelConstructor("data/target_sentences.txt",
#                        "data/results_fillers_RTs.csv", advanced=True)
