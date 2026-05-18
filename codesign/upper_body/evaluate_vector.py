import argparse
import numpy as np

from codesign.upper_body.arm_evaluation import ArmEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one upper-body co-design vector."
    )
    parser.add_argument(
        "--dx",
        nargs="+",
        type=float,
        required=True,
        help="Design vector values (6D shoulder-only or 8D including elbow terms).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display meshcat/optimization debug output during evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dx = np.array(args.dx, dtype=float)
    evaluator = ArmEvaluator()
    value = evaluator.evaluate(dx, disp=args.display)
    print(f"cost={value}")


if __name__ == "__main__":
    main()
