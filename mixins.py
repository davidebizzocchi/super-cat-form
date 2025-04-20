
from typing import List, Type

from cat.log import log

from cat.plugins.super_cat_form.super_cat_form_events import FormEvent
from cat.plugins.super_cat_form.super_cat_form import SuperCatForm, form_tool
from cat.plugins.super_cat_form.utils import format_class_name


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


class InsideFormMixin:
    """
    Mixin that enables a form to contain and manage sub-forms.
    
    This mixin provides functionality to associate and manage sub-forms within
    a parent form, including automatic tool generation for each sub-form.
    """

    # List of sub-forms associated with this form
    sub_forms: List[Type[SuperCatForm]] = []

    def __init__(self, *args, **kwargs):
        """
        Initialize the mixin and associate sub-forms.
        """
        self.assoc_sub_forms(self.sub_forms)
        super().__init__(*args, **kwargs)

    @classmethod
    def assoc_sub_forms(cls, sub_forms: List[Type[SuperCatForm]]) -> None:
        """
        Associate sub-forms with the current form and create corresponding tools.
        
        This method:
        1. Validates that all sub-forms are proper SuperCatForm subclasses
        2. Ensures each sub-form has the SubFormMixin functionality
        3. Creates and attaches tools for starting each sub-form
        
        Args:
            sub_forms (List[Type[SuperCatForm]]): List of sub-forms to associate
            
        Raises:
            ValueError: If any sub-form is not a subclass of SuperCatForm
        """
        for form in sub_forms:
            # Validate form type
            if not issubclass(form, SuperCatForm):
                raise ValueError(f"All sub-forms must be subclasses of SuperCatForm")

            # Ensure form has SubFormMixin functionality
            if not issubclass(form, SubFormMixin):
                form = type(form, (SubFormMixin, form), {})

            # Create a tool for starting the sub-form
            def recall_sub_form_tool(self):
                return self.start_sub_form(form)

            # Set tool metadata
            recall_sub_form_tool.__name__ = format_class_name(f"{form.name}_tool")
            recall_sub_form_tool.__doc__ = f"Collect the {form.name} information"

            # Create and attach the form tool
            recall_method = form_tool(recall_sub_form_tool, return_direct=True, examples=form.start_examples)
            setattr(cls, recall_sub_form_tool.__name__, recall_method)

    def start_sub_form(self, form_class: Type[SuperCatForm]) -> str:
        """
        Create and activate a new form, saving this form as the parent form.
        
        Args:
            form_class (Type[SuperCatForm]): The form class to instantiate
            
        Returns:
            str: The initial message from the new form
        """
        # Create the new form instance
        new_form = form_class(self.cat)

        # Set the parent form reference
        new_form.parent_form = self

        # Activate the new form
        self.cat.working_memory.active_form = new_form

        log.debug(f"Started sub-form: {new_form.name} from parent: {self.name}")

        # Return the first message of the new form
        return new_form.next()["output"]
