import pyactr as actr
import random


# TODO:
#    - For some reason the word "met" is not retrieved when using subsymbolic
#      mode. Maybe retrieval of words is not always successful when attempted
#      in subsymbolic mode??
#    - A way to stop the retrieval of the current read word when retrieving a
#      reference

class Model:

    NSEC_IN_YEAR = int(round(60 * 60 * 24 * 364.25))

    def __init__(self, gui=True, subsymbolic=False, activation_trace=False):
        self.gui = gui
        # Left-to-right reading. Start at (0, 0)

        self.environment = actr.Environment(focus_position=(0, 0))

        if subsymbolic:
            self.model = actr.ACTRModel(environment=self.environment,
                                        automatic_visual_search=False,
                                        activation_trace=activation_trace,
                                        emma_noise=False,
                                        subsymbolic=True,
                                        retrieval_threshold=0.92,
                                        instantaneous_noise=1.77,
                                        latency_factor=0.45,
                                        latency_exponent=0.28,
                                        decay=0.095,
                                        motor_prepared=True,
                                        eye_mvt_scaling_parameter=0.23)
        else:
            self.model = actr.ACTRModel(environment=self.environment,
                                        automatic_visual_search=False,
                                        eye_mvt_scaling_parameter=0.23,
                                        motor_prepared=True,
                                        emma_noise=False)

        self.lexicon = ["de", "besprak", "met", "het", "onderzoeksvoorstel",
                        "die", "periode", "geen", "nieuwe",
                        "resultaten", "van", "periode", "een"]

        self.nouns = [("professor", "masc"), ("vriend", "fem")]

        self.object_indicators = ["enkele"]

        self.back_reference_objects = [("hij", "masc"), ("zij", "fem")]

        self.sentence_terminators = ["."]

        actr.chunktype("word", "form, cat")
        actr.chunktype("noun", "form, cat, gender")

        self.chunks = []

        self.chunks += [actr.makechunk(typename="word", nameofchunk=w, form=w,
                                       cat="terminator")
                        for w in self.sentence_terminators]

        self.chunks += [actr.makechunk(typename="word", nameofchunk=w, form=w,
                                       cat="word")
                        for w in self.lexicon]

        self.chunks += [actr.makechunk(typename="word",
                                       nameofchunk=w, form=w,
                                       cat="object_indicator")
                        for w in self.object_indicators]

        self.chunks += [actr.makechunk(typename="noun", nameofchunk=n, form=n,
                                       cat="noun", gender=g)
                        for (n, g) in self.nouns + self.back_reference_objects]

        for c in self.chunks:
            for _ in range(0, self.freq(c.form)):
                self.model.decmem.add(c, time=random.randint(-self.NSEC_IN_YEAR
                                                             * 15, 0))

        actr.chunktype("goal", ("state, expecting_object,"
                                "first_word_attended, subject_attended,"
                                "in_second_sentence"))
        self.model.goal.add(actr.makechunk(nameofchunk="start",
                                           typename="goal", state="start",
                                           expecting_object=False,
                                           first_word_attended=False,
                                           subject_attended=False,
                                           in_second_sentence=False))

        self.model.visualBuffer("visual", "visual_location", self.model.decmem,
                                # finst=float("inf"))
                                finst=5)

        # Find a word on the screen
        # TODO: maybe change to lowest
        self.model.productionstring(name="find word", string="""
            =g>
            isa goal
            state 'start'
            ?visual_location>
            buffer empty
            ==>
            ?visual_location>
            attended False
            +visual_location>
            isa _visuallocation
            screen_x lowest
            screen_y lowest
        """)

        # Attend to the object found
        self.model.productionstring(name="attend word", string="""
            =g>
            isa goal
            state 'start'
            =visual_location>
            isa _visuallocation
            ?visual>
            state free
            ==>
            =g>
            isa goal
            state 'encoding'
            +visual>
            isa _visual
            cmd move_attention
            screen_pos =visual_location
            ~visual_location>
        """)

        self.add_parse_rules()

    # encoding -> encoding_done
    def add_parse_rules(self):
        # Dummy recall.
        self.model.productionstring(name="waiting for word", string="""
            =g>
            isa goal
            state 'encoding'
            =visual>
            isa _visual
            value "___"
            ==>
            =g>
            isa goal
            state 'start'
        """)

        # Recall the first word.
        # This is a special case because we want this to set the
        # first_word_attended and we don't want this rule to fire when a
        # subject is expected.
        self.model.productionstring(name="recalling (first word)", string="""
            =g>
            isa goal
            state 'encoding'
            expecting_object ~True
            first_word_attended False
            ?visual>
            buffer full
            =visual>
            isa _visual
            value =val
            value ~"___"
            value ~None
            ==>
            =g>
            isa goal
            state 'encoding_done'
            first_word_attended True
            +retrieval>
            isa word
            form =val
        """)

        # Recall an object. An object is expected because an object indicator
        # has just been read.
        self.model.productionstring(name="recalling (object expected)", string="""
            =g>
            isa goal
            state 'encoding'
            expecting_object True
            ?visual>
            buffer full
            =visual>
            isa _visual
            value =val
            value ~"___"
            value ~None
            ==>
            =g>
            isa goal
            state 'encoding_done'
            expecting_object False
            +retrieval>
            isa noun
            form =val
        """)

        # Recall a subject. A subject is expected because the first word has
        # been read and no subject has yet to be attended.
        # (we assume that the subject is alsways the second word of a sentence)
        self.model.productionstring(name="recalling (subject expected)", string="""
            =g>
            isa goal
            state 'encoding'
            first_word_attended True
            subject_attended False
            ?visual>
            buffer full
            =visual>
            isa _visual
            value =val
            value ~"___"
            value ~None
            ==>
            =g>
            isa goal
            state 'encoding_done'
            subject_attended True
            +retrieval>
            isa noun
            form =val
        """)

        # A normal word recall that should be fired if no other recall can be
        # fired.
        self.model.productionstring(name="recalling", string="""
            =g>
            isa goal
            state 'encoding'
            expecting_object ~True
            subject_attended True
            first_word_attended True
            ?visual>
            buffer full
            =visual>
            isa _visual
            value =val
            value ~"___"
            value ~None
            ==>
            =g>
            isa goal
            state 'encoding_done'
            +retrieval>
            isa word
            form =val
        """)

        # XXX: might not be realistic.
        # The problem is that when a visual object dissapears from the
        # environment while it was found (and therefore being attended), the
        # value of the visual buffer will be set to None.
        # We have to make sure we try to find a new word when this happens
        # to find a new word.
        #
        # But it would be more natural for the value to just become the word
        # that was originally at the place of the dashes.
        self.model.productionstring(name="recover from lost word", string="""
            =g>
            isa goal
            state 'encoding'
            ?visual>
            state free
            =visual>
            isa _visual
            value None
            ==>
            =g>
            isa goal
            state 'start'
        """)

        # TODO: could replace the cat ~ stuf with cat word if word is the only
        # thing we are  catching here in the end
        self.model.productionstring(name="lexeme retrieved", string="""
            =g>
            isa goal
            state 'encoding_done'
            ?retrieval>
            buffer full
            state free
            =retrieval>
            isa word
            cat ~noun
            cat ~terminator
            cat ~object_indicator
            ==>
            =g>
            isa goal
            state 'start'
            +manual>
            isa _manual
            cmd press_key
            key 'space'
        """)

        # Retrieve a noun without trying to recall a reference to a previous
        # sentence (no sentence read before)
        self.model.productionstring(name="lexeme retrieved (noun)", string="""
            =g>
            isa goal
            state 'encoding_done'
            in_second_sentence ~True
            ?retrieval>
            buffer full
            state free
            =retrieval>
            isa noun
            cat noun
            ==>
            =g>
            isa goal
            state 'start'
            +manual>
            isa _manual
            cmd press_key
            key 'space'
        """)

        self.model.productionstring(name=("lexeme retrieved (noun):"
                                          " start reference retrieval"),
                                    string="""
            =g>
            isa goal
            state 'encoding_done'
            in_second_sentence True
            ?retrieval>
            buffer full
            state free
            =retrieval>
            isa noun
            cat noun
            gender =g
            ==>
            =g>
            isa goal
            state 'retrieving_reference'
            +retrieval>
            isa noun
            attended True
        """)

        self.model.productionstring(name="lexeme retrieved (terminator)", string="""
            =g>
            isa goal
            state 'encoding_done'
            ?retrieval>
            buffer full
            state free
            =retrieval>
            isa word
            cat terminator
            ==>
            =g>
            isa goal
            state 'start'
            in_second_sentence True
            +manual>
            isa _manual
            cmd press_key
            key 'space'
        """)

        self.model.productionstring(name="lexeme retrieved (object indicator)", string="""
            =g>
            isa goal
            state 'encoding_done'
            ?retrieval>
            buffer full
            state free
            =retrieval>
            isa word
            cat object_indicator
            ==>
            =g>
            isa goal
            state 'start'
            expecting_object True
            +manual>
            isa _manual
            cmd press_key
            key 'space'
        """)

        self.model.productionstring(name="no lexeme found", string="""
            =g >
            isa goal
            state 'encoding_done'
            ?retrieval>
            state error
            ==>
            =g>
            isa goal
            state 'start'
        """)

        self.model.productionstring(name="reference retrieved",
                                    string="""
            =g>
            isa goal
            state 'encoding_done'
            in_second_sentence True
            ?retrieval>
            buffer full
            state free
            =retrieval>
            isa word
            ==>
            =g>
            isa goal
            state 'start'
            !retrieval>
            show form
            +manual>
            isa _manual
            cmd press_key
            key 'space'
        """)

        self.model.productionstring(name="no reference retrieved:",
                                    string="""
            =g>
            isa goal
            state 'encoding_done'
            in_second_sentence True
            ?retrieval>
            state error
            ==>
            =g>
            isa goal
            state 'start'
        """)

    def freq(self, _):
        return 1000

    def sentence_to_stimuli(self, s):
        wl = s.split(' ')

        return ([{p: self.env_word(wl, p), p+1: self.env_dash(p+1)}
                for p in range(len(wl) - 1)] +
                [{len(wl)-1: self.env_word(wl, len(wl)-1)}])

    def sentence_to_stimuli2(self, s):
        wl = s.split(' ')
        return [dict(enumerate(self.dashed_env(wl, p)))
                for p in range(len(wl))]

    TEXT_MARGIN = (30, 10)
    TEXT_SPACING = (60, 12)

    def dashed_env(self, wl, p):
        pos = list(self.TEXT_MARGIN)
        max_s = self.environment.size
        for (i, w) in enumerate(wl):
            yield {'text': w if i == p else '_' * len(w),
                   'position': tuple(pos)}
            pos[0] += self.TEXT_SPACING[0]
            if pos[0] > max_s[0]:
                pos[0] = self.TEXT_MARGIN[0]
                pos[1] += self.TEXT_SPACING[1]

    def env_pos(self, p):
        pos = list(self.TEXT_MARGIN)
        max_s = self.environment.size
        x = ((p * self.TEXT_SPACING[0]) %
             (max_s[0] - 2 * self.TEXT_MARGIN[0])) + pos[0]
        y = int((p * self.TEXT_SPACING[0]) /
                (max_s[0] - 2 * self.TEXT_MARGIN[0])) \
            * self.TEXT_SPACING[1] + self.TEXT_MARGIN[1]
        return (x, y)

    def env_word(self, wl, p):
        w = wl[p]
        return {'text': w, 'position': self.env_pos(p)}

    def env_dash(self, p):
        return {'text': '___', 'position': self.env_pos(p)}

    def sim(self):
        s = ("de professor besprak met geen enkele vriend de nieuwe resultaten"
             " die periode . hij besprak")
        # The simulation requires a dictionary for some reason...
        # w = dict(enumerate(self.sentence_to_stimuli(s)))
        # print(w)
        sim = self.model.simulation(
            realtime=True,
            gui=self.gui,
            environment_process=self.environment.environment_process,
            stimuli=self.sentence_to_stimuli(s),
            triggers=['space'],
            # Set the timeout to something big enough so that timeout will
            # hopefully never trigger. The stimuli should always be cycled by
            # the triggers instead.
            times=1000)
        return sim

    def run(self):
        # while True:
        #     sim.step()
        #     if sim.current_event.action.startswith("RETRIEVED:"):
        #         print("{}: {}".format(sim.current_event.time,
        #                               sim.current_event.action))
        # sim.run()
        self.sim().run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", help="Don't run the model",
                        action="store_true")
    parser.add_argument("-g", "--gui", help="Run with a gui",
                        action="store_true")
    parser.add_argument("-t", "--activation-trace",
                        help="Output activation trace", action="store_true")

    # XXX: For now nargs="+" is fine. If we also need positional arguments the
    # grammar of the arguments will be ambiguous. Use action="append" then.
    # (This is also a bit better since this is the GNU standard way I think?)
    parser.add_argument("-f", "--filters", nargs="+",
                        help=("Filter the output of the simulation. Specify "
                              "one or more strings as filters. A step of the "
                              "simulation is only printed when it contains "
                              "such string"))
    parser.add_argument("-s", "--subsymbolic",
                        help="Use the subsymbolic ACT-R model",
                        action="store_true")

    parser.add_argument("--diff-formatted",
                        help="Output the steps in a way that helps diff.",
                        action="store_true")

    args = parser.parse_args()

    m = Model(gui=args.gui, subsymbolic=args.subsymbolic,
              activation_trace=args.activation_trace)
    sim = m.sim()
    if not args.dry_run and not args.filters and not args.diff_formatted:
        sim.run()
    elif args.filters:
        while True:
            sim.step()
            if True in map(lambda x: x in sim.current_event.action,
                           args.filters):
                print("{}: {}".format(sim.current_event.time,
                                      sim.current_event.action))
    elif args.diff_formatted:
        while True:
            sim.step()
            print(sim.current_event.action)
