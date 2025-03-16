from cat.experimental.form import CatForm
from pydantic import BaseModel
from .super_cat_form import SuperCatForm

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from cat.plugins.super_cat_form.prompts import DEFAULT_HUMAN_READABLE_PROMPT
from cat.plugins.super_cat_form.super_cat_form_events import FormEvent

from cat.log import log
import pprint

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
    log.error(f"Form data: {form_data}, form name: {self.name}, form active: {self.active_form}")
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
    
    @staticmethod
    def _create_single_field_models(base_model: type[BaseModel]) -> dict[str, type[BaseModel]]:

        field_models = {}
        for field_name, field_info in base_model.model_fields.items():
            log.error(f"Field name: {field_name}, field info: {field_info}, {type(field_info)}")
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
        self.events.on(FormEvent.FORM_INITIALIZED, self.create_step_forms)

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

    def create_step_forms(self, context):
        self._field_models = self._create_single_field_models(self.model_getter())
        log.error(f"Field models: {pprint.pformat(self._field_models)}")
        self._form_classes = {}

        field_names = list(self._field_models.keys())
        log.error(f"Field names: {field_names}")

        last_form = None
        for i in range(len(field_names)):
            field_name = field_names[i]
            field_model, field_info = self._field_models[field_name]

            log.error(f"\n\nField name: {field_name}, field model: {field_model}, field info: {field_info}, last form: {last_form.name if last_form else None}")
            last_form = self._create_form_class(field_model, field_info, last_form)
            self._form_classes[field_name] = last_form

        log.error(f"Form classes: {pprint.pformat(self._form_classes)}")
        log.error(f"First form: {last_form}, type: {last_form.name}")

        # Now start the first form and specify need set it
        self.first_form: SuperCatForm = last_form(
            cat=self.cat,
            parent_form=self,
        )
        self.__is_first_form_set = False

    def _create_form_class(self, model_class: type[BaseModel], field_info, next_form=None) -> SuperCatForm:
        form_name = self.format_class_name(f"{model_class.__name__}Form")
        return type(
            form_name,
            (SuperCatForm,),
            {
                "model_class": model_class,
                "ask_confirm": True,
                "description": field_info.description or f"{model_class.__name__} form",
                "next_form": next_form,
                "name": form_name,
                "submit": base_submit,
            }
        )

    def _get_field_model(self, field_name: str) -> type[BaseModel]:
        return self._field_models.get(field_name)
    
    def _get_field_model_instance(self, field_name: str, field_data: dict) -> BaseModel:
        field_model = self._get_field_model(field_name)
        return field_model(**field_data)