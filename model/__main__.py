from model import Model

import argparse
import csv
import numpy as np

from trainer import Trainer
from measurer import Measurer, EventMeasurer, EventIntervalMeasurer


def process_sentence_pairs(l):
    for s in l:
        st = map(str.strip, s.split("."))
        yield list(map(lambda x: x.split(" "), st))


def run_args(p):
    parser = argparse.ArgumentParser(parents=[p])
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
    return parser.parse_args()


def run_mode(parser):
    args = run_args(parser)

    s = [("de professor besprak met geen enkele vriend de nieuwe resultaten"
          " die periode. hij besprak")]

    sl = list(process_sentence_pairs(s))

    lex = ["de", "besprak", "met", "het", "onderzoeksvoorstel",
           "die", "periode", "geen", "nieuwe",
           "resultaten", "van", "periode", "een"]
    nouns = [("professor", "masc"), ("vriend", "masc")]

    object_indicators = ["enkele"]

    back_reference_objects = [("hij", "masc"), ("zij", "fem")]

    m = Model(sl, lex, nouns=nouns + back_reference_objects,
              object_indicators=object_indicators,
              gui=args.gui, subsymbolic=args.subsymbolic,
              activation_trace=args.activation_trace,
              advanced=not args.basic_mode)
    sim = m.sim()

    print(m.model.decmem)

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


def train_args(p):
    parser = argparse.ArgumentParser(parents=[p])
    parser.add_argument("-n", "--noun-mode",
                        help="Train the advanced (noun) model",
                        metavar='BASIC_RESULTS_FILENAME')
    parser.add_argument("sentence_file", help=("The file containing the"
                                               " sentences"))
    parser.add_argument("rt_file", help=("The file containing the reaction"
                                         " times"))

    parser.add_argument("-r", "--calculate-results",
                        metavar="RESULTS_FILENAME",
                        help=("Calculate results from a csv containing"
                              " parameter values"))

    parser.add_argument("-f", "--filter-mode",
                        help=("Set the filter mode."
                              " By default allow-all is used"),
                        choices=["allow-all", "allow-mis-match"],
                        default="allow-all"),

    parser.add_argument("-p", "--plot",
                        help=("Plot differences if -r, --calculate-results"
                              "is specified"),
                        action="store_true")

    args = parser.parse_args()

    if args.plot and not args.calculate_results:
        raise Warning("-p, --plot won't do anything unless -r,"
                      " --calculate-results is specified")

    return args


def read_from_csv(filename):
    with open(filename, 'r') as f:
        r = csv.reader(f)
        results = {row[0]: float(row[1]) for row in r}
    return results


def train_mode(parser):
    args = train_args(parser)
    model_args = {"gui": True, "subsymbolic": True}
    if args.noun_mode:
        m = EventIntervalMeasurer(("RULE SELECTED: lexeme "
                                   "retrieved (noun): "
                                   "start reference retrieval"),
                                  ("RULE FIRED: reference"
                                   " retrieved"), True)
        p = ["latency_factor",
             "latency_exponent",
             "intercept"]

        default_param_values = read_from_csv(args.noun_mode)

    else:
        m = EventMeasurer("KEY PRESSED: SPACE", True)
        p = ["eye_mvt_scaling_parameter",
             "eye_mvt_angle_parameter",
             "rule_firing"]

        default_param_values = {}

    t = Trainer(args.sentence_file,
                args.rt_file, m, p,
                model_args=model_args, verbose=True,
                default_param_values=default_param_values,
                advanced=bool(args.noun_mode))

    if args.noun_mode:
        if args.filter_mode == "allow-all":
            t.model_constructor.entry_type_filters = \
                [lambda x: x[3] == ["een", "mis", "match"] or
                 x[3] == ["een", "match", "mis"]]
            t.model_constructor.noun_filters = \
                [lambda x: x["Subject_Gender"] != x["Object_Gender"]]
        else:
            t.model_constructor.entry_type_filters = \
                [lambda x: x[3] == ["een", "mis", "match"]]

            t.model_constructor.noun_filters = \
                [lambda x: x["Subject_Gender"] == "mis" and
                 x["Object_Gender"] == "match"]

    if args.calculate_results:
        results = read_from_csv(args.calculate_results)
        r = t.collect_results(**results)
        print("Y - Yobs:")
        print(r - t.observed_measures())
        print()
        print("Max error of {}".format(
            np.max(np.abs(r - t.observed_measures()))))
        print("Mean error of {}.".format(
            np.mean(np.abs(r - t.observed_measures()))))

        if args.plot:
            t.plot(**results)
    else:
        t.train()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("mode", help='The mode of the program.',
                    choices=["run", "train"])

args, _ = parser.parse_known_args()
if args.mode == "train":
    train_mode(parser)
elif args.mode == "run":
    run_mode(parser)
