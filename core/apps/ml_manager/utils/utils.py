import json
import re

class Utils:
    @staticmethod
    def natural_sort_key(s):
        """
        Function to generate a natural sort key.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    @staticmethod
    def print_metrics(metrics, epoch_samples, phase):
        outputs = [f"{k}: {metrics[k] / epoch_samples:.4f}" for k in metrics.keys()]
        print(f"{phase}: {', '.join(outputs)}")

    @staticmethod
    def parse_list_or_value(value):
        try:
            # Spróbuj przekonwertować wartość na JSON
            parsed_value = json.loads(value)
            # Jeśli wartość nie jest listą, zwróć ją jako listę
            if not isinstance(parsed_value, list):
                return [parsed_value]
            return parsed_value
        except json.JSONDecodeError:
            # Jeśli wartość nie jest poprawnym JSON, zwróć ją jako listę
            return [value]

    