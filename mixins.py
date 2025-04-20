
from cat.log import log

from cat.plugins.super_cat_form.super_cat_form_events import FormEvent
from cat.plugins.super_cat_form.super_cat_form import SuperCatForm


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


class SubFormMixin:
    """
    A mixin that allows a form to be used as a subform.
    
    This mixin enables form nesting by tracking the parent form and handling
    the restoration of the parent form when the subform is closed or submitted.
    """

    # Track the form that started this form (if any)
    parent_form: SuperCatForm = None

    def _setup_default_handlers(self) -> None:
        super()._setup_default_handlers()

        # Add handler for form exit to restore previous form
        self.events.on(FormEvent.FORM_CLOSED, self._restore_parent_form)
        self.events.on(FormEvent.FORM_SUBMITTED, self._restore_parent_form)

    def _restore_parent_form(self, *args, **kwargs) -> None:
        """
        Restore parent form when this form is closed or submitted.
        
        This method is called when the form is closed or submitted to ensure
        the parent form becomes active again in the working memory.
        """
        if self.parent_form is not None:
            self.cat.working_memory.active_form = self.parent_form
            log.debug(f"Restored previous form: {self.parent_form.name}")
