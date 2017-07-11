from model import Model

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

parser.add_argument("-b", "--basic-mode",
                    help="Run the model in basic mode (reading only)",
                    action="store_true")

args = parser.parse_args()

m = Model(gui=args.gui, subsymbolic=args.subsymbolic,
          activation_trace=args.activation_trace,
          advanced=not args.basic_mode)
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
