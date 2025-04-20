import re


def format_class_name(name):
    """
    Formats a class name into snake_case by inserting underscores before capital letters
    and converting the result to lowercase.

    Args:
        name (str): The class name to format.

    Returns:
        str: The formatted name in snake_case.
    """
    def replacement(match):
        """Helper function to determine the replacement for regex matches."""
        # If the match is from the second pattern (?<!^)(?=[A-Z]), insert an underscore
        if match.group(0) == '':
            return '_'
        # If the match is from the first pattern ([A-Z]+)([A-Z][a-z]), insert underscore between groups
        return match.group(1) + '_' + match.group(2)

    # Regex pattern to handle
    pattern = r'([A-Z]+)([A-Z][a-z])|(?<!^)(?=[A-Z])'

    # Apply the regex substitution, convert to lowercase, and replace spaces with underscores
    return re.sub(pattern, replacement, name).lower().replace(" ", "_")