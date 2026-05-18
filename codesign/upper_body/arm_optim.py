import numpy as np
from cmaes import CMA

from codesign.upper_body.arm_evaluation import ArmEvaluator

PRINT_EACH_CANDIDATE = False


if __name__ == "__main__":
    evaluate = ArmEvaluator()
    dx = np.zeros(6, dtype=float)
    optimizer = CMA(mean=np.float64(dx), sigma=15)
    best_value = None
    best_x = None

    for generation in range(60):
        solutions = []
        generation_best_value = None
        generation_best_x = None
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = evaluate.evaluate(x)
            solutions.append((x, np.float64(value)))
            if generation_best_value is None or value < generation_best_value:
                generation_best_value = value
                generation_best_x = np.array(x, dtype=float)
            if best_value is None or value < best_value:
                best_value = value
                best_x = np.array(x, dtype=float)
            if PRINT_EACH_CANDIDATE:
                print(f"#{generation} value={value} x={x}")
        optimizer.tell(solutions)
        print(
            f"#{generation} best_generation_value={generation_best_value} "
            f"best_generation_x={generation_best_x}"
        )

    print("best_value=", best_value)
    print("best_x=", best_x)
