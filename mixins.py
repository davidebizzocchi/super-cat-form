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
