from typing import Any, Dict, Type

from cat.plugins.super_cat_form.super_cat_form_events import FormEvent, FormEventContext
from cat.plugins.super_cat_form.super_cat_form import SuperCatForm


class ContextWorkflowMixin:
    """
    Mixins that provides workflow capabilities to other mixins that don't implement them.
    """

    def get_sub_form_kwargs(self, form_class: Type[SuperCatForm]) -> Dict[str, Any]:
        """
        Get initialization parameters for sub-forms.
        
        Ensures the workflow is properly passed to child forms.
        """
        return {
            "cat": self.cat,
            "workflow": self.workflow,
        }

    def check_start_sub_form(self, new_form_instance: SuperCatForm) -> bool:
        """
        Check if a sub-form can be started based on its dependencies.
        """
        if hasattr(new_form_instance, "form_action"):
            if not self.workflow.can_execute(new_form_instance.form_action):
                return self.message()["output"]

        return super().check_start_sub_form(new_form_instance)

