import re
import json
import csv
from model import Model

from itertools import product


def format_word(w):
    return w.strip().replace('.', '').lower()


def flatten(l):
    for i in l:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        else:
            yield i


class ModelConstructor():
    def __init__(self, sentence_filepath,
                 word_rt_csv, advanced=False,
                 **kwargs):
        """
        Arguments:
            sentence_filepath: path to the txt file containing the sentences in
                               correct format
            word_rt_csv: path to the csv file containing the reaction time
                           data
            advanced: weather or not the constructor should generate advanced
                      models. When the ModelConstructor is in advanced mode,
                      sentence_filepath is assumed to contain noun gender
                      match/mismatch data and word_rt_csv is assumed to contain
                      RTs of nouns. When ModelContructor is in non-advanced
                      mode, sentence_filepath should contain word RT info
                      instead.
            kwargs: all kwargs are passed to the models constructed by the
                    ModelConstructor.
        """
        self.kwargs = kwargs
        self.entries = []
        self.advanced = advanced
        self.read_sentences(sentence_filepath, self.advanced)
        self.word_rt_csv = word_rt_csv
        self.sentence_filepath = sentence_filepath

        # A list of filters applied to the noun csv entries in advanced mode
        self.noun_filters = []
        # self.noun_filters = [lambda x: x["Subject_Gender"] !=
        #                      x["Object_Gender"]]

        # self.noun_filters = [lambda x: x["Subject_Gender"] == "mis" and
        #                      x["Object_Gender"] == "match"]

        # A list of filters applied to the list of entries in advanced mode
        self.entry_type_filters = []
        # self.entry_type_filters = [lambda x: x[3] == ["een", "mis", "match"] or
        #                            x[3] == ["een", "match", "mis"]]
        # self.entry_type_filters = [lambda x: x[3] == ["een", "mis", "match"]]
        el = len(self.entries)
        print("entries successfully read: {}/{}"
              .format(el, self.num_total_entries))

    def parse(self):
        el = len(self.entries)
        skipped = None
        if el != self.num_total_entries:
            print("some entries were skipped, figuring out which ones...")
            skipped = self.skipped_entries()

        if self.advanced:
            for f in self.entry_type_filters:
                self.entries = filter(f, self.entries)
            self.entries = list(self.entries)
            print("successful entries after filters {}/{}".format(len(self.entries),
                                                             el))

        if self.advanced:
            self.parse_noun_rt_csv(self.word_rt_csv, skipped)
            print("All noun rt information found")
        else:
            self.parse_rt_csv(self.word_rt_csv)
            print("All rt information found")

    def skipped_entries(self):
        l = set((tuple(flatten((x[4], tuple(x[3])))) for x in self.entries))
        full_set = set(product(range(1, 33), product(["geen", "een"],
                                                     product(["mis", "match"],
                                                             ["mis", "match"])
                                                     )))
        full_set = set(map(lambda x: tuple(flatten(x)), full_set))
        return full_set - l

    def read_sentences(self, fn, vc):
        # TODO: also retrieve second sentence
        def rem_dot(wl):
            if wl[-1][-1] == ".":
                wl[-1] = wl[-1][:-1]
            return wl

        def parse_sentence(l, i):
            if self.advanced:
                ss, nl, idx, t = self.lex_sentence(l)
            else:
                ss, nl = self.lex_sentence(l)
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
            if self.advanced:
                return sl, nl, indicator, t, idx
            else:
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

    def parse_rt_csv(self, fn):
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
                                               float(r['RT']) / 1000)
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
                              "RTs!")

            # self.rts = {format_word(r['Word']): r['RT'] for r in reader}

    def parse_noun_rt_csv(self, fn, skipped=None):
        with open(fn, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')

            e = list(reader)
            for f in self.noun_filters:
                e = filter(f, e)
            e = sorted(e, key=lambda x: int(x['Item']))

            # Filter out any skipped entries
            # XXX: for now we assume all words are from "een" sentences (this
            # info is not provided by the csv)
            if skipped:
                e = list(filter(lambda x: (int(x['Item']), "een",
                                           x['Subject_Gender'],
                                           x['Object_Gender']) not in skipped,
                                e))

            if len(e) != len(self.entries):
                print(len(e), len(self.entries))
                raise Warning("number of found RTs does not match"
                              "number of entries!")
            self.rt_list = [float(r['RT']) / 1000 for r in e]

    def model_generator(self, **kwargs):
        def gender(match_type, sentence):
                match_gender = "masc" if "hij" in wl[1] else "fem"
                mis_gender = "masc" if match_gender == "fem" else "fem"
                d = {"match": match_gender, "mis": mis_gender}
                return [d[m] for m in match_type]

        if not self.advanced:
            for wl, nl in self.entries:
                wl = [[w for w, _ in l] for l in wl]
                lex = set([w for l in wl for w in l])
                yield Model([wl], lex, advanced=self.advanced,
                            **self.kwargs, model_params=kwargs)
        else:
            bro = [("hij", "masc"), ("zij", "fem")]
            for wl, nl, ind, t, _, in self.entries:
                old_nl = nl
                nl = list(zip(nl, gender(t[1:], wl)))
                wl = [[w for w in l] for l in wl]
                lex = set([w for l in wl for w in l]) \
                    - (set(old_nl) | set(["hij", "zij"]) | set([ind]))

                yield Model([wl], lex, advanced=self.advanced,
                            object_indicators=[ind],
                            nouns=nl + bro,
                            **self.kwargs, model_params=kwargs)

    def rts(self):
        if self.advanced:
            for e in self.rt_list:
                yield e
        else:
            for e in self.entries:
                for l in e[0]:
                    for _, f in l:
                        yield f

    def lex_sentence(self, l):
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

        if self.advanced:
            sentence_id = o[0][1]
            sentence_type = o[0][0].split('_')[:3]

            # Swap mismatch data to match assumed (subject, object) order.
            # For some reason the tags provided in the file are of the form:
            # object_subject_id
            tmp = sentence_type[2]
            sentence_type[2] = sentence_type[1]
            sentence_type[1] = tmp

            return (s, nl, sentence_id, sentence_type)
        else:
            return (s, nl)


# mc = ModelConstructor("data/fillers.txt",
#                       "data/results_fillers_RTs.csv", advanced=False)
# mc2 = ModelConstructor("data/target_sentences.txt",
#                        "data/results_fillers_RTs.csv", advanced=True)
