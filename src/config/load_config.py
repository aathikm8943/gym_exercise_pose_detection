import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.rules.bicep_curl_rule import BicepCurlRules
from src.rules.lateral_raise_rule import LateralRaiseRules

# Map each label to its corresponding rule class
RULE_CLASSES = {
    "Bicep Curl": BicepCurlRules,
    "Lateral Raise": LateralRaiseRules
}

def load_rule_evaluators(config_path: str):
    """
    Load exercise rule evaluators from a YAML config file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: A dictionary {exercise_name: evaluator_instance}
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    evaluators = {}
    for exercise_name in config.get("exercises", []):
        rule_class = RULE_CLASSES.get(exercise_name)
        if rule_class is None:
            print(f"Warning: No rule class found for exercise '{exercise_name}'. Skipping.")
            continue
        evaluators[exercise_name] = rule_class()

    return evaluators
