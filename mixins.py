from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from cat.log import log

from cat.plugins.super_cat_form.super_cat_form import SuperCatForm
from cat.plugins.super_cat_form.prompts import DEFAULT_HUMAN_READABLE_PROMPT
from cat.plugins.super_cat_form.super_cat_form_events import FormEvent


class HumanFriendlyInteractionsMixin:
    def _generate_base_message(self):
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
    form_result = self.form_data_validated

    if form_result is None:
        return {
            "output": "Invalid form data"
        }
    
    return {
        "output": f"{form_result}"
    }
    
class StepByStepMixin:

    __is_first_form_set = False

    default_submit = base_submit
    
    @staticmethod
    def _create_single_field_models(base_model: type[BaseModel]) -> dict[str, type[BaseModel]]:

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
        super()._setup_default_handlers()

        self.events.on(
            FormEvent.FORM_INITIALIZED,
            self.create_step_forms
        )

        self.events.on(
            FormEvent.INSIDE_FORM_CLOSED,
            self.fill_model_with_previous_values
        )

    def next(self):
        # Set the first form
        if not self.__is_first_form_set:
            self.__is_first_form_set = True

            # Modify the active form
            self.active_form = self.first_form

            # Initialize this form
            super().next()

            # Return the new form initialiazed
            return self.first_form.next()
        
        return super().next()
    
    def fill_model_with_previous_values(self, context):
        
        prev_form_names = list(self.prev_results.keys())
        prev_form_values = list(self.prev_results.values())

        # Check the form that trigger event, is a next_form and not a inside_form
        if not context.form_id in prev_form_names:
            return

        log.debug(f"[EVENT: _on_previous_form_inactive] {self.name} previous form is inactive (by next_form)")
        
        results = {}
        for field_name, field_value in zip(prev_form_names, prev_form_values):
            results[field_name] = getattr(field_value, field_name, None)

        model = self.model_getter().model_validate(results)

        # for field_name, field_value in zip(prev_form_names, prev_form_values):
        #     if hasattr(model, field_name):
        #         setattr(model, field_name, field_value)
        
        old_model = self._model.copy() if self._model is not None else {}
        self._model = model.model_dump()

        # emit events for updated fields
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
        self._field_models = self._create_single_field_models(self.model_getter())
        self._form_classes = {}

        field_names = list(self._field_models.keys())

        last_form = None
        for i in range(len(field_names)):
            field_name = field_names[i]
            field_model, field_info = self._field_models[field_name]

            last_form = self._create_form_class(field_name, field_model, field_info, last_form)
            self._form_classes[field_name] = last_form

        # Now start the first form and specify need set it
        self.first_form: SuperCatForm = last_form(
            cat=self.cat,
            parent_form=self,
        )
        self.__is_first_form_set = False

    def _create_form_class(self, form_name, model_class: type[BaseModel], field_info, next_form=None) -> SuperCatForm:
        form_class_name = self.format_class_name(f"{model_class.__name__}Form")
        return type(
            form_class_name,
            (SuperCatForm,),
            {
                "model_class": model_class,
                "ask_confirm": True,
                "description": field_info.description or f"{model_class.__name__} form",
                "next_form": next_form,
                "name": form_name,
                "submit": self.default_submit,
            }
        )

    def _get_field_model(self, field_name: str) -> type[BaseModel]:
        return self._field_models.get(field_name)
    
    def _get_field_model_instance(self, field_name: str, field_data: dict) -> BaseModel:
        field_model = self._get_field_model(field_name)
        return field_model(**field_data)