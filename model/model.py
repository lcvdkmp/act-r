import pyactr as actr
import random

# Store subject in goal
# Context activation
# Enable subsymbolic


class Model:

    NSEC_IN_YEAR = int(round(60 * 60 * 24 * 365.25))

    TEXT_MARGIN = (30, 30)
    TEXT_SPACING = (60, 12)

    def __init__(self, sentence_list, lexicon, nouns=[], object_indicators=[],
                 gui=True, subsymbolic=False, activation_trace=False,
                 advanced=True):
        self.gui = gui

        self.sentence_pairs = sentence_list
        self.lexicon = lexicon

        vx = self.env_pos_x_virtual(self.env_size()) + self.TEXT_MARGIN[0]
        vy = 320

        # Set initial focus on the first word
        self.environment = actr.Environment(focus_position=self.TEXT_MARGIN,
                                            size=(vx, vy))

        self.stimuli = list(self.stimuli_gen())
        self.nouns = nouns
        self.object_indicators = object_indicators
        # TODO: estimate:
        #       - eye_mvt
        #           half normal sd=0.5
        #       - eye_map
        #           gamma as=0.7
        #       - rule_firing (with and without)
        #           half normal sd=0.5

        if subsymbolic:
            self.model = actr.ACTRModel(environment=self.environment,
                                        automatic_visual_search=False,
                                        activation_trace=activation_trace,
                                        emma_noise=False,
                                        subsymbolic=True,
                                        # retrieval_threshold=0.92,
                                        # instantaneous_noise=1.77,
                                        # latency_factor=0.45,
                                        # latency_exponent=0.28,
                                        # decay=0.095,
                                        motor_prepared=True
                                        # eye_mvt_scaling_parameter=0.23)
                                        )
        else:
            self.model = actr.ACTRModel(environment=self.environment,
                                        automatic_visual_search=False,
                                        eye_mvt_scaling_parameter=0.23,
                                        motor_prepared=True,
                                        emma_noise=False)

        # self.imaginal = self.model.set_goal(name="imaginal")

        actr.chunktype("word", "form, cat")
        actr.chunktype("noun", "form, cat, gender")
        # actr.chunktype("read_word", "word, use, gender")

        self.chunks = []

        self.chunks += [actr.makechunk(typename="word", nameofchunk=w, form=w,
                                       cat="word")
                        for w in self.lexicon]

        self.chunks += [actr.makechunk(typename="word",
                                       nameofchunk=w, form=w,
                                       cat="object_indicator")
                        for w in self.object_indicators]

        self.chunks += [actr.makechunk(typename="noun", nameofchunk=n, form=n,
                                       cat="noun", gender=g)
                        for (n, g) in self.nouns]

        for c in self.chunks:
            # for _ in range(0, self.freq(c.form)):
            #     self.model.decmem.add(c, time=random.randint(-self.NSEC_IN_YEAR
            #                                                  * 15, 0))
            self.model.decmem.add(c)

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
                                finst=5)

        # Find a word on the screen
        self.model.productionstring(name="find word", string="""
            =g>
            isa goal
            state 'start'
            ?visual_location>
            buffer empty
            ==>
            =g>
            isa goal
            state 'attend'
            ?visual_location>
            attended False
            +visual_location>
            isa _visuallocation
            screen_x lowest
            screen_y lowest
        """)

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

        # XXX: might not be realistic.
        # The problem is that when a visual object dissapears from the
        # environment while it was found (and therefore being attended), the
        # value of the visual buffer will be set to None.
        # We have to make sure we try to find a new word when this happens.
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
            ~retrieval>
        """)

        if advanced:
            self.advanced_mode()
        else:
            self.basic_mode()

    def basic_mode(self):
        self.model.productionstring(name="attend word", string="""
            =g>
            isa goal
            state 'attend'
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

        self.model.productionstring(name="recalling", string="""
            =g>
            isa goal
            state 'encoding'
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
            ~visual>
        """)

        self.model.productionstring(name="lexeme retrieved", string="""
            =g>
            isa goal
            state 'encoding_done'
            ?retrieval>
            buffer full
            state free
            =retrieval>
            isa word
            ==>
            =g>
            isa goal
            state 'start'
            +manual>
            isa _manual
            cmd press_key
            key 'space'
            ~retrieval>
        """)

    def advanced_mode(self):

        # Attend to the object found. There is a distinction between words on
        # the first and second sentence to update the in_second_sentence state.
        #
        # Note that we know that words in the first sentence have a y position
        # of self.TEXT_MARGIN[0]
        self.model.productionstring(name="attend word (first sentence)", string="""
            =g>
            isa goal
            state 'attend'
            in_second_sentence ~True
            =visual_location>
            isa _visuallocation
            screen_y {}
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
        """.format(self.TEXT_MARGIN[0]))

        self.model.productionstring(name="attend word (first sentence - reset state)", string="""
            =g>
            isa goal
            state 'attend'
            in_second_sentence True
            =visual_location>
            isa _visuallocation
            screen_y {}
            ?visual>
            state free
            ==>
            =g>
            isa goal
            state 'encoding'
            in_second_sentence False
            subject_attended False
            first_word_attended False
            expecting_object False
            +visual>
            isa _visual
            cmd move_attention
            screen_pos =visual_location
            ~visual_location>
        """.format(self.TEXT_MARGIN[0]))

        self.model.productionstring(name="attend word (second sentence)", string="""
            =g>
            isa goal
            state 'attend'
            =visual_location>
            isa _visuallocation
            screen_y ~{}
            ?visual>
            state free
            ==>
            =g>
            isa goal
            state 'encoding'
            in_second_sentence True
            +visual>
            isa _visual
            cmd move_attention
            screen_pos =visual_location
            ~visual_location>
        """.format(self.TEXT_MARGIN[0]))

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
            ~visual>
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
            +retrieval>
            isa noun
            form =val
            ~visual>
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
            +retrieval>
            isa noun
            form =val
            ~visual>
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
            ~visual>
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
            cat ~object_indicator
            ==>
            =g>
            isa goal
            state 'start'
            +manual>
            isa _manual
            cmd press_key
            key 'space'
            ~retrieval>
        """)

        # Retrieve a noun without trying to recall a reference to a previous
        # sentence (no sentence read before)
        # TODO: is the object subject split needed? Or could we just say "role
        # noun" or something

        # TODO:
        # =retrieval>
        # store something that it was attended. role=subject or something
        self.model.productionstring(name="lexeme retrieved (object)", string="""
            =g>
            isa goal
            state 'encoding_done'
            in_second_sentence ~True
            expecting_object True
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
            expecting_object False
            =retrieval>
            role object
            +manual>
            isa _manual
            cmd press_key
            key 'space'
            ~retrieval>
        """)

        self.model.productionstring(name="lexeme retrieved (subject)", string="""
            =g>
            isa goal
            state 'encoding_done'
            in_second_sentence ~True
            first_word_attended True
            subject_attended False
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
            subject_attended True
            =retrieval>
            role subject
            +manual>
            isa _manual
            cmd press_key
            key 'space'
            ~retrieval>
        """)

        # TODO: find a way to cath both role subject and role object
        # Current approach doesn't work

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
            gender =gd
            ==>
            =g>
            isa goal
            state 'retrieving_reference'
            +retrieval>
            isa noun
            gender =gd
            role subject
            role object
            ~retrieval>
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
            ~retrieval>
        """)

        self.model.productionstring(name="reference retrieved",
                                    string="""
            =g>
            isa goal
            state 'retrieving_reference'
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
            ~retrieval>
        """)

        self.model.productionstring(name="no reference retrieved:",
                                    string="""
            =g>
            isa goal
            state 'retrieving_reference'
            in_second_sentence True
            ?retrieval>
            state error
            ==>
            =g>
            isa goal
            state 'start'
            ~retrieval>
        """)

    def freq(self, _):
        return 1000

    def stimuli_gen(self):
        for s in self.sentence_pairs:
            def special_product(l1, l2):
                """
                Given a list l2 and a list of indices l1, yield all tuples
                (x, y) where x is an index in l1 and y is l2[x].
                """
                for i in l1:
                    for j in range(0, len(l2[i])):
                        yield (i, j)

            # Calculate the positions of the words in the first and the second
            # sentence.
            pos = [(x * self.TEXT_SPACING[0] + self.TEXT_MARGIN[0],
                    self.TEXT_MARGIN[1] + i * self.TEXT_SPACING[1])
                   for (i, x) in special_product((0, 1), s)]

            # Yield stimuli environments for all words.
            for i, p in enumerate(pos):
                # If p[1] = self.TEXT_MARGIN[1] the corresponding word came
                # from s[0]. The word came from s[1] otherwise.
                if p[1] is self.TEXT_MARGIN[1]:
                    w = s[0][i]
                else:
                    w = s[1][i - (len(s[0]))]

                d = [{"text": w,
                      "position": p,
                      "vis_delay": len(w)}]

                if i < len(pos)-1:
                    d += [{"text": "___", "position": pos[i+1],
                           "vis_delay": 1}]

                # The simulation requires a dictionary for some reason...
                yield dict(enumerate(d))

    def sentence_to_stimuli(self, s):
        wl = s.split(" ")

        return ([{p: self.env_word(wl, p), p+1: self.env_dash(p+1)}
                for p in range(len(wl) - 1)] +
                [{len(wl)-1: self.env_word(wl, len(wl)-1)}])

    def env_size(self):
        return max(map(lambda x: max(map(len, x)), self.sentence_pairs))

    def env_pos_x_virtual(self, p):
        return self.TEXT_MARGIN[0] + self.TEXT_SPACING[0] * p

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
        # Note the vis_delay. This makes the reading delay proportional to the
        # word length
        return {'text': w, 'position': self.env_pos(p), 'vis_delay': len(w)}

    def env_dash(self, p):
        return {'text': '___', 'position': self.env_pos(p), 'vis_delay': 3}

    def sim(self):
        # The simulation requires a dictionary for some reason...
        # w = dict(enumerate(self.sentence_to_stimuli(s)))
        # print(w)

        # If gui is set to False, the simulation only runs for at most 1
        # second. Setting realtime to False and gui to True results in the
        # expected effect when no gui is desired.
        sim = self.model.simulation(
            realtime=self.gui,
            gui=True,
            environment_process=self.environment.environment_process,
            stimuli=self.stimuli,
            triggers=['space'],
            # Set the timeout to something big enough so that timeout will
            # hopefully never trigger. The stimuli should always be cycled by
            # the triggers instead.
            times=1000)
        return sim

    def run(self):
        self.sim().run()
