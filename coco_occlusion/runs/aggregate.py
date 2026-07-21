import json
import os
import glob
import statistics

# Configuration
BASE_DIR = "."  # Change this to the root directory containing run1, run2, run3
# TASK_DIR = "identify_occluded_object"
OCCLUSION_LEVELS = [
    "no",
    "low",
    "medium",
    "high",
]

def collect_results(base_dir):
    """
    Traverse all run directories and collect accuracy values per model and occlusion level.
    Returns a dict: { model_name: { occlusion_level: [acc1, acc2, ...] } }
    """
    model_data = {}

    run_dirs = sorted(glob.glob(os.path.join(base_dir, "run*")))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under '{base_dir}'")

    for run_dir in run_dirs:
        task_path = os.path.join(run_dir)         # , TASK_DIR
        if not os.path.isdir(task_path):
            print(f"Warning: Task directory not found: {task_path}, skipping.")
            continue

        model_dirs = glob.glob(os.path.join(task_path, "*"))
        if not model_dirs:
            print(f"Warning: No result files or folders found in {task_path}, skipping.")
            continue

        for model_dir in model_dirs:
            # model_name = os.path.basename(json_file).replace("_results.json", "")
            model_name = os.path.basename(model_dir)
            json_file = os.path.join(model_dir, f"{model_name}_results.json")

            with open(json_file, "r") as f:
                data = json.load(f)

            overall = data.get("overall", {})

            if model_name not in model_data:
                model_data[model_name] = {level: [] for level in OCCLUSION_LEVELS}

            for level in OCCLUSION_LEVELS:
                if level in overall:
                    accuracy = overall[level].get("accuracy")
                    if accuracy is not None:
                        model_data[model_name][level].append(accuracy)
                    else:
                        print(f"Warning: 'accuracy' missing in {json_file} -> overall.{level}")
                else:
                    print(f"Warning: '{level}' missing in {json_file}")

    return model_data


def compute_statistics(model_data):
    """
    Compute mean and standard deviation for each model and occlusion level.
    """
    output = {}

    for model_name, levels in sorted(model_data.items()):
        output[model_name] = {}
        for level in OCCLUSION_LEVELS:
            values = levels.get(level, [])
            if len(values) == 0:
                mean_acc = None
                sd_acc = None
            elif len(values) == 1:
                mean_acc = round(values[0], 4)
                sd_acc = 0.0
            else:
                mean_acc = round(statistics.mean(values), 4)
                sd_acc = round(statistics.stdev(values) / (len(values) ** 0.5), 4)  # sample std dev

            print(f"\nModel: {model_name}\nLevel: {level}")
            output[model_name][level] = {
                "mean_accuracy": mean_acc,
                "sd_accuracy": sd_acc,
            }
            print(f"Output: {output[model_name][level]}")
            try:
                print(f"Std dev not over mean: {statistics.stdev(values)}")
            except:
                print("Cannot calculate std dev.")

    return output


def main():
    model_data = collect_results(BASE_DIR)

    if not model_data:
        print("No model data found. Please check BASE_DIR and directory structure.")
        return

    output = compute_statistics(model_data)

    output_file = "aggregated_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Results written to '{output_file}'")
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()