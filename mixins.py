
import inspect
from typing import Any, Dict, List, Optional, Type, Tuple
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from cat.log import log
from cat.experimental.form import CatFormState

from cat.plugins.super_cat_form.super_cat_form_events import FormEvent, FormEventContext
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


class NextFormMixin(FreshStartMixin):
    """
    Mixin that allows to open a new form after the current form is closed.
    
    This mixin enables form chaining by specifying a next form to open
    when the current form is closed or submitted.
    """

    next_form: SuperCatForm = None
    show_next_form_message: bool = True

    def _setup_default_handlers(self) -> None:
        super()._setup_default_handlers()

        self.events.on(FormEvent.FORM_CLOSED, self.start_next_form)
        self.events.on(FormEvent.FORM_SUBMITTED, self.start_next_form)

        self.events.on(FormEvent.FORM_INITIALIZED, self.initialize_as_next_form)

    def initialize_as_next_form(self, *args, **kwargs) -> None:
        """Initialize the form as a next form."""
        self.modify_user_message()

        if inspect.isclass(self.next_form):
            self.next_form = self.next_form(self.cat)

    def start_next_form(self, *args, **kwargs) -> None:
        """Open the next form after the current form is closed."""
        if self.next_form is not None:
            self.cat.working_memory.active_form = self.next_form
            self.modify_user_message()

    def next(self):
        """
        Override the next method to handle form chaining.
        
        Returns:
            The next form's output or the current form's output
        """
        output = super().next()

        if self._state == CatFormState.CLOSED and self.show_next_form_message:
            self.cat.send_chat_message(output["output"], save=False)
            return self.next_form.next()

        return output


class StepByStepMixin:
    """
    A mixin that converts a single complex form into multiple sequential single-field forms,
    allowing users to complete forms step by step.
    
    By default step-forms use FreshStartMixin, so their are independent from the conversation.
    """

    __is_first_form_initialized: bool = False
    _field_models: Dict[str, Tuple[Type[BaseModel], FieldInfo]] = {}
    _form_classes: Dict[str, Type[SuperCatForm]] = {}

    first_form: SuperCatForm = None
    first_form_class: Type[SuperCatForm] = None
    step_forms_base_class: Type | tuple[Type, ...] = (FreshStartMixin,)

    @staticmethod
    def field_form_submit(self, form_data: dict) -> dict:
        """
        Default submission handler for forms.
        
        Args:
            form_data: The data submitted through the form.
            
        Returns:
            dict: Response containing the output message.
        """
        form_result = self.form_data_validated

        if form_result is None:
            return {"output": "Invalid form data"}

        self.main_form.form_data[self.field_name] = getattr(form_result, self.field_name, None)
        return {"output": f"{form_result}"}
    
    @staticmethod
    def _create_single_field_models(base_model: Type[BaseModel]) -> Dict[str, Tuple[Type[BaseModel], FieldInfo]]:
        """
        Creates individual Pydantic models for each field in the base model.
        
        For each field in base_model, returns a dict with the field name as key,
        and a tuple containing a new BaseModel with only this field and the field info.
        
        Args:
            base_model: The original Pydantic model to split into single fields
            
        Returns:
            Dict[str, Tuple[Type[BaseModel], FieldInfo]]: Mapping of field names to (model, field_info) tuples
        """
        field_models = {}
        for field_name, field_info in base_model.model_fields.items():
            field_models[field_name] = (
                type(
                    f"{base_model.__name__}_{field_name}",
                    (BaseModel,),
                    {
                        "__annotations__": {field_name: field_info.annotation},
                        field_name: field_info
                    }
                ),
                field_info
            )
        return field_models

    def _setup_default_handlers(self) -> None:
        """Set up the default event handlers for the step-by-step form workflow."""
        super()._setup_default_handlers()
        self.events.on(FormEvent.FORM_INITIALIZED, self.initialize_step_forms)

    def initialize_step_forms(self, context: FormEventContext) -> None:
        """Initialize the step forms."""

        if self.step_forms_base_class is None:
            raise ValueError("step_forms_base_class must be set")
        if not isinstance(self.step_forms_base_class, tuple):
            self.step_forms_base_class = (self.step_forms_base_class,)

        self.create_step_forms(context)
        self.prepare_next_step_iteration()

    def create_step_forms(self, context: FormEventContext) -> None:
        """
        Creates individual step forms for each field in the model.
        Triggered on FORM_INITIALIZED event.
        
        Args:
            context: Event context containing form initialization data
        """
        self._field_models = self._create_single_field_models(self.model_getter())
        self._form_classes = {}

        field_names = list(self._field_models.keys())
        last_form = self

        # Create step form classes with one field each and link them in sequence
        for field_name in field_names:
            field_model, field_info = self._field_models[field_name]
            last_form = self._create_form_class(
                field_name=field_name,
                model_class=field_model,
                field_info=field_info,
                next_form=last_form
            )
            self._form_classes[field_name] = last_form

        # Initialize the first form to start the sequence
        # Last form created is the first form to start
        self.first_form_class = last_form

    def next(self):
        """
        Override the standard next method to handle step-by-step form initialization.
        
        Returns:
            The next form's output or the current form's output
        """
        if not self.__is_first_form_initialized:
            self.__is_first_form_initialized = True

            # Modify the active form
            self.cat.working_memory.active_form = self.first_form

            # Initialize this form
            super().next()

            # Return the new form initialized
            return self.first_form.next()
        
        return super().next()

    def prepare_next_step_iteration(self):
        """
        Prepare the form for a new iteration of the step-by-step process.
        
        This method:
        1. Resets the form initialization flag to allow starting a new iteration
        2. Creates a new instance of the first form in the sequence
        """
        self.__is_first_form_initialized = False
        self.first_form = self.first_form_class(self.cat)
    
    def get_step_form_kwargs(
            self,
            form_name: str,
            model_class: Type[BaseModel], 
            field_info: FieldInfo,
            field_name: str, 
            next_form: Optional[SuperCatForm] = None
        ) -> Dict[str, Any]:
        """
        Generate keyword arguments for creating a step form class.
        
        Args:
            form_name: Name of the form (usually the field name)
            model_class: The single-field Pydantic model for this step
            field_info: Field metadata from the original model
            field_name: Name of the field being processed
            next_form: The next form in the sequence
            
        Returns:
            Dict[str, Any]: Keyword arguments for creating the form class
        """
        return {
            "model_class": model_class,
            "ask_confirm": True,
            "description": field_info.description or f"{model_class.__name__} form",
            "next_form": next_form,
            "name": form_name,
            "submit": self.field_form_submit,
            "main_form": self,
            "field_name": field_name,
        }

    def _create_form_class(
            self,
            field_name: str,
            model_class: Type[BaseModel], 
            field_info: FieldInfo,
            next_form: Optional[SuperCatForm] = None
        ) -> Type[SuperCatForm]:
        """
        Create a single step form class for one field.
        
        Args:
            field_name: Name of the form/field
            model_class: The Pydantic model for this single field
            field_info: Field metadata
            next_form: The next form in the sequence
            
        Returns:
            Type[SuperCatForm]: A new form class for this step
        """
        form_name = f"{model_class.__name__}Form"

        return type(
            format_class_name(form_name),
            (NextFormMixin, *self.step_forms_base_class, SuperCatForm),
            self.get_step_form_kwargs(
                form_name=form_name,
                model_class=model_class,
                field_info=field_info,
                field_name=field_name,
                next_form=next_form
            )
        )


class NotCloseMixin:
    """
    Mixins that avoid the form can be closed.
    The FORM_SUBMITTED and FORM_CLOSED events are triggered anyway.
    """

    def next(self):
        """Avoid that CatFormState.CLOSED is returned and set CatFormState.INCOMPLETE instead."""

        output = super().next()
        if self._state == CatFormState.CLOSED:
            self._state = CatFormState.INCOMPLETE
        return output

    def check_exit_intent(self):
        """The form cannot be closed, so this method must always return False."""
        return False
