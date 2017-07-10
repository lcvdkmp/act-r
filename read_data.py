import re
import json
import csv


def format_word(w):
    return w.strip().replace('.', '').lower()


class ModelConstructor():
    def __init__(self, sentence_filepath,
                 word_freq_csv, validity_check=True):
        self.entrances = []
        self.read_sentences(sentence_filepath, validity_check)
        print("Entrances successfully parsed: {}/{}"
              .format(len(self.entrances), self.num_total_entrances))

        self.parse_freq_csv(word_freq_csv)

        self.check_freq_completeness()
        print("All frequency information found")

    def read_sentences(self, fn, vc):
        def parse_sentence(l, i):
            ss, nl = lex_sentence(l)
            wl = ss[0].split(" ")
            if wl[-1][-1] == ".":
                wl[-1] = wl[-1][:-1]
            wl = list(map(format_word, wl))
            nl = list(map(str.strip, nl))
            if not vc:
                return wl, nl
            il = []
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
            return wl, nl, indicator

        self.num_total_entrances = 0
        with open(fn, 'r') as f:
            for i, l in enumerate(f):
                self.num_total_entrances += 1
                try:
                    self.entrances += [parse_sentence(l, i + 1)]
                except Warning as w:
                    print("Warning:", w)
                    continue

    def parse_freq_csv(self, fn):
        with open(fn, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            self.freqs = {format_word(r['Word']): r['RT'] for r in reader}

    def check_freq_completeness(self):
        ist = set([w for (ss, _) in self.entrances for w in ss])
        ist -= set(self.freqs.keys())
        if len(ist) > 0:
            raise Exception("Missing frequency information for:"
                            "{}".format(ist))


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


mc = ModelConstructor("data/fillers.txt",
                      "data/results_fillers_RTs.csv", validity_check=False)
# mc2 = ModelConstructor("data/target_sentences.txt",
#                        "data/results_fillers_RTs.csv", validity_check=False)
