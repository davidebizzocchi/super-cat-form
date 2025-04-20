class FreshStartMixin:
    """
    Mixin that initializes forms as independent conversations.

    This mixin provides functionality to start a form with a clean conversation history,
    effectively treating each form as a new conversation. This is useful when you want
    to isolate form interactions from previous conversation context.
    """

    # Flag for cleaning up conversation history - each form is a completely new conversation
    fresh_start: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.fresh_start:
            self.clear_history()

    def clear_history(self, index: int = -1) -> None:
        """
        Clear the conversation history, optionally keeping the last message.
        
        Args:
            index (int): Index of the message to keep. Defaults to -1 (last message).
        """
        self.cat.working_memory.history = self.cat.working_memory.history[index:]

    def empty_history(self) -> None:
        """Clear the entire conversation history."""
        self.cat.working_memory.history = []

    def modify_user_message(self, text: str = "") -> None:
        """
        Modify the current user message.
        
        Args:
            text (str): New text for the user message. Defaults to empty string.
        """
        self.cat.working_memory.user_message_json.text = text

