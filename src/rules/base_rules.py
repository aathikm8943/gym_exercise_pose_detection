# pose_form_checker/rules/base_rules.py

from abc import ABC, abstractmethod

class BaseRuleSet(ABC):
    @abstractmethod
    def evaluate_all(self):
        """Evaluates all the defined rules and returns a dictionary of results."""
        pass
