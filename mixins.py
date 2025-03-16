from pydantic import BaseModel
from pydantic.fields import FieldInfo

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from cat.log import log

from cat.plugins.super_cat_form.super_cat_form import SuperCatForm
from cat.plugins.super_cat_form.prompts import DEFAULT_HUMAN_READABLE_PROMPT
from cat.plugins.super_cat_form.super_cat_form_events import FormEvent


class HumanFriendlyInteractionsMixin:
    """
    A mixin class that enhances form interactions with more human-friendly messages.
    Uses LLM to transform standard messages into more conversational ones.
    """
    def _generate_base_message(self):
        """
        Generates a human-friendly version of the base message using an LLM.
        
        Returns:
            str: The human-friendly message.
        """
        message = super()._generate_base_message()

        prompt = PromptTemplate(
            template=DEFAULT_HUMAN_READABLE_PROMPT,
            input_variables=["message"]
        )
        parser = StrOutputParser()

        chain = prompt | self.cat._llm | parser
        return chain.invoke({
            "message": message
        })


def base_submit(self, form_data):
    """
    Default submission handler for forms.
    
    Args:
        form_data: The data submitted through the form.
        
    Returns:
        dict: Response containing the output message.
    """
    form_result = self.form_data_validated

    if form_result is None:
        return {
            "output": "Invalid form data"
        }
    
    return {
        "output": f"{form_result}"
    }
    
    
class StepByStepMixin:
    """
    A mixin that converts a single complex form into multiple sequential single-field forms,
    allowing users to complete forms step by step.
    """

    __is_first_form_set = False

    default_submit = base_submit
    
    @staticmethod
    def _create_single_field_models(base_model: type[BaseModel]) -> dict[str, tuple[type[BaseModel], FieldInfo]]:
        """
        Creates individual Pydantic models for each field in the base model.
        
        For each field in base_model, returns a dict with the field name as key,
        and a tuple containing a new BaseModel with only this field and the field info.
        
        Args:
            base_model: The original Pydantic model to split into single fields
            
        Returns:
            dict: Mapping of field names to (model, field_info) tuples
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
                field_info)

        return field_models

    def _setup_default_handlers(self):
        """
        Sets up the default event handlers for the step-by-step form workflow.
        Extends the parent's default handlers.
        """
        super()._setup_default_handlers()

        self.events.on(
            FormEvent.FORM_INITIALIZED,
            self.create_step_forms
        )

        self.events.on(
            FormEvent.INSIDE_FORM_CLOSED,
            self.populate_model_from_steps
        )

    def populate_model_from_steps(self, context):
        """
        Fills the model with values from previously completed step forms.
        Triggered when an inside form is closed.
        
        Args:
            context: Event context containing form information
        """
        prev_form_names = list(self.prev_results.keys())
        prev_form_values = list(self.prev_results.values())

        # Check if the form that triggered the event is a next_form and not an inside_form
        if not context.form_id in prev_form_names:
            return

        log.debug(f"[EVENT: _on_previous_form_inactive] {self.name} previous form is inactive (by next_form)")
        
        results = {}
        for field_name, field_value in zip(prev_form_names, prev_form_values):
            results[field_name] = getattr(field_value, field_name, None)

        model = self.model_getter().model_validate(results)
        
        old_model = self._model.copy() if self._model is not None else {}
        self._model = model.model_dump()

        # Emit events for updated fields
        updated_fields = {
            k: v for k, v in self._model.items()
            if k not in old_model or old_model[k] != v
        }

        if updated_fields:
            self.events.emit(
                FormEvent.FIELD_UPDATED,
                {
                    "fields": updated_fields,
                    "old_values": {k: old_model.get(k) for k in updated_fields}
                },
                self.name
            )

    def create_step_forms(self, context):
        """
        Creates individual step forms for each field in the model.
        Triggered on FORM_INITIALIZED event.
        
        Args:
            context: Event context
        """
        self._field_models = self._create_single_field_models(self.model_getter())
        self._form_classes = {}

        field_names = list(self._field_models.keys())

        # Create step form classes with one field each and link them in sequence
        last_form = None
        for i in range(len(field_names)):
            field_name = field_names[i]
            field_model, field_info = self._field_models[field_name]

            last_form = self._create_form_class(field_name, field_model, field_info, last_form)
            self._form_classes[field_name] = last_form

        # Initialize the first form to start the sequence
        # Last form created is the first form to start
        self.first_form: SuperCatForm = last_form(
            cat=self.cat,
            parent_form=self,
        )

        # Mark that we need to initialize the form after CheshireCat completes initialization
        self.__is_first_form_set = False

    def next(self):
        """
        Overrides the standard next method to handle step-by-step form initialization.
        
        Returns:
            The next form in the sequence
        """
        # Set up the first form if not already done
        if not self.__is_first_form_set:
            self.__is_first_form_set = True

            # Modify the active form
            self.active_form = self.first_form

            # Initialize this form
            super().next()

            # Return the new form initialized
            return self.first_form.next()
        
        return super().next()
    
    def get_step_form_kwargs(self, form_name, model_class: type[BaseModel], field_info, next_form=None) -> dict:
        """
        Generates the keyword arguments for creating a step form class.
        
        Args:
            form_name: Name of the form (usually the field name)
            model_class: The single-field Pydantic model for this step
            field_info: Field metadata from the original model
            next_form: The next form in the sequence
            
        Returns:
            dict: Kwargs for creating the form class
        """
        return {
            "model_class": model_class,
            "ask_confirm": True,
            "description": field_info.description or f"{model_class.__name__} form",
            "next_form": next_form,
            "name": form_name,
            "submit": self.default_submit,
        }

    def _create_form_class(self, form_name, model_class: type[BaseModel], field_info, next_form=None) -> SuperCatForm:
        """
        Creates a single step form class for one field.
        
        Args:
            form_name: Name of the form/field
            model_class: The Pydantic model for this single field
            field_info: Field metadata
            next_form: The next form in the sequence
            
        Returns:
            type: A new form class for this step
        """
        return type(
            self.format_class_name(f"{model_class.__name__}Form"),
            (SuperCatForm,),
            self.get_step_form_kwargs(form_name, model_class, field_info, next_form)
        )
