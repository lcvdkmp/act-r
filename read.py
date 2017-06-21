import pyactr as actr


class Model:
    def __init__(self):
        # Left-to-right reading. Start at (0, 0)
        self.environment = actr.Environment(focus_position=(0, 0))

        self.model = actr.ACTRModel(environment=self.environment,
                                    automatic_visual_search=False)

        actr.chunktype("word", "form")

        self.lexicon = list(map(str, range(60)))
        for w in self.lexicon:
            self.model.decmem.add(actr.makechunk(typename="word", form=w))

        actr.chunktype("goal", "state")
        self.model.goal.add(actr.makechunk(nameofchunk="start",
                                           typename="goal", state="start"))

        # Note finst=50. This determines how long the memory is about what
        # objects were atteded to.
        self.model.visualBuffer("visual", "visual_location", self.model.decmem,
                                finst=50)

        # Find a word on the screen
        # TODO: currently the left-to-right reading depends heavily on atteded.
        # If a very long string of text is presented to the model, it will,
        # after a while, start reading the first word again. This is due to the
        # fact that finst in finite and therefore the memory of attended words
        # is finite as well. (Note that the model will search for words closest
        # to (0, 0) that are not yet attended with x precedence)
        #
        # A better model would be to restrict the newline search from searching
        # for above the current line.
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
            state 'retrieving'
            +visual>
            isa _visual
            cmd move_attention
            screen_pos =visual_location
            ~visual_location>
        """)

        self.model.productionstring(name="recalling", string="""
            =g>
            isa goal
            state 'retrieving'
            =visual>
            isa _visual
            value =val
            ==>
            =g>
            isa goal
            state 'retrieval_done'
            +retrieval>
            isa word
            form =val
        """)

        self.model.productionstring(name="lexeme retrieved", string="""
            =g>
            isa goal
            state 'retrieval_done'
            ?retrieval>
            buffer full
            state free
            ==>
            =g>
            isa goal
            state 'start'
            +manual>
            isa _manual
            cmd press_key
            key 'space'
        """)

        self.model.productionstring(name="no lexeme found", string="""
            =g>
            isa goal
            state 'retrieval_done'
            ?retrieval>
            state error
            ==>
            =g>
            isa goal
            state 'start'
            +manual>
            isa _manual
            cmd press_key
            key 'space'
        """)

    def sentence_to_env(self, s):
        margin = (30, 10)
        pos = list(margin)
        spacing = (60, 12)
        max_s = self.environment.size
        for w in s.split(' '):
            yield {'text': w, 'position': tuple(pos)}
            pos[0] += spacing[0]
            if pos[0] > max_s[0]:
                pos[0] = margin[0]
                pos[1] += spacing[1]

    def run(self):
        s = " ".join(self.lexicon)
        w = dict(enumerate(self.sentence_to_env(s)))
        sim = self.model.simulation(
            realtime=True,
            gui=True,
            environment_process=self.environment.environment_process,
            stimuli=w,
            triggers='',
            times=1)

        while True:
            sim.step()
            if sim.current_event.action.startswith("RETRIEVED:"):
                print(sim.current_event.action)


if __name__ == "__main__":
    m = Model()
    m.run()
