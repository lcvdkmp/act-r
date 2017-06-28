import pyactr as actr


class Model:
    def __init__(self, gui=True):
        self.gui = gui
        # Left-to-right reading. Start at (0, 0)
        self.environment = actr.Environment(focus_position=(0, 0))

        self.model = actr.ACTRModel(environment=self.environment,
                                    automatic_visual_search=False)

        actr.chunktype("word", "form")

        self.lexicon = list(map(str, range(80)))
        for w in self.lexicon:
            self.model.decmem.add(actr.makechunk(typename="word", form=w))

        actr.chunktype("goal", "state")
        self.model.goal.add(actr.makechunk(nameofchunk="start",
                                           typename="goal", state="start"))

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

        self.dummy_parse()

    # encoding -> encoding_done
    def dummy_parse(self):
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
        """)

        # XXX: might not be realistic.
        # The problem is that when a visual object dissapears from the
        # environment while it was found (and therefore being attended), the
        # value of the visual buffer will be set to None.
        # We have to make sure we try to find a new word when this happens
        # to find a new word.
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

        self.model.productionstring(name="lexeme retrieved", string="""
            =g>
            isa goal
            state 'encoding_done'
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
            state 'endoding_done'
            ?retrieval>
            state error
            ==>
            =g>
            isa goal
            state 'start'
        """)

    # encoding -> encoding_done
    def parse(self):
        pass

    def sentence_to_env(self, s):
        wl = s.split(' ')

        return [{p: self.env_word(wl, p), p+1: self.env_dash(p+1)}
                for p in range(len(wl))]

    def sentence_to_env2(self, s):
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
        s = " ".join(self.lexicon)
        # The simulation requires a dictionary for some reason...
        # w = dict(enumerate(self.sentence_to_env(s)))
        # print(w)
        sim = self.model.simulation(
            realtime=True,
            gui=self.gui,
            environment_process=self.environment.environment_process,
            stimuli=self.sentence_to_env(s),
            triggers=['space'],
            times=10000)
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
    args = parser.parse_args()

    m = Model(gui=args.gui)
    sim = m.sim()
    if not args.dry_run:
        sim.run()
