import argparse
import numpy as np

from codesign.hip.hip_optim import EvaluateRobot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one design vector with the hip co-design cost functions."
    )
    parser.add_argument(
        "--dx",
        nargs="+",
        type=float,
        required=True,
        help="Design vector values (6 or 7 floats).",
    )
    parser.add_argument(
        "--mode",
        choices=["walk", "sidewalk", "combined", "visualize"],
        default="combined",
        help="Evaluation mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dx = np.array(args.dx, dtype=float)
    evaluator = EvaluateRobot()

    if args.mode == "walk":
        value = evaluator.evaluate_walk(dx, vizual=False)
        print(f"walk_cost={value}")
        print(f"walk_velocity_penalty={evaluator.velocity}")
    elif args.mode == "sidewalk":
        value = evaluator.evaluate_sidewalk(dx, vizual=False)
        print(f"sidewalk_cost={value}")
        print(f"sidewalk_velocity_penalty={evaluator.velocity}")
    elif args.mode == "combined":
        value = evaluator.evaluate(dx)
        print(f"combined_cost={value}")
    else:
        evaluator.vizalize(dx)


if __name__ == "__main__":
    main()
