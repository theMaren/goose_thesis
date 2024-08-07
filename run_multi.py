""" Main search driver. """
import argparse
import os
#from planners.search import search_cmd

_DOWNWARD_CPU = "./planners/downward_cpu/fast-downward.py"
_DOWNWARD_GPU = "./planners/downward_gpu/fast-downward.py"

def get_cmd(args):

    aux_file = args.aux_file
    plan_file = args.plan_file
    mf = os.path.abspath(args.model_path)
    df = os.path.abspath(args.domain_pddl)
    pf = os.path.abspath(args.problem_pddl)
    time_limit = args.overall_time_limit

    description = repr(hash(repr(args))).replace("-", "n")

    if aux_file is None:
        os.makedirs("aux", exist_ok=True)
        aux_file = f"aux/{description}.aux"

    if plan_file is None:
        os.makedirs("plans", exist_ok=True)
        plan_file = f"plans/{description}.plan"

    h_goose = f'goose(model_path="{mf}", domain_file="{df}", instance_file="{pf}", model_type = "gnn")'

    
    fd_search = f"eager_greedy([{h_goose},ff()])"
    cmd = cmd = f"{_DOWNWARD_CPU} --overall-time-limit {time_limit} --sas-file {aux_file} --plan-file {plan_file} {df} {pf} --search '{fd_search}'"
    
    cmd = f"export GOOSE={os.getcwd()}/learner && {cmd}"

    return cmd, aux_file



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "domain_pddl", type=str, help="path to domain pddl file"
    )
    parser.add_argument(
        "problem_pddl", type=str, help="path to problem pddl file"
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="path to saved model weights",
    )

    parser.add_argument(
        "--algorithm",
        "-s",
        type=str,
        default="eager",
        choices=["eager"],
        help="solving algorithm using the heuristic",
    )
    parser.add_argument(
        "--aux-file",
        type=str,
        default=None,
        help="path of auxilary file such as *.sas or *.lifted",
    )
    parser.add_argument(
        "--plan-file",
        type=str,
        default=None,
        help="path of *.plan file",
    )

    parser.add_argument(
        "--overall-time-limit",
        type=str,
        default="1200s",
        help="Overall time limit",
    )

    args = parser.parse_args()


    cmd, aux_file = get_cmd(args)
    print("Executing the following command:")
    print(cmd)
    print()
    os.system(cmd)
    if os.path.exists(aux_file):
        os.remove(aux_file)
