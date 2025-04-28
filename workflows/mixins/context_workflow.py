from typing import Any, Callable, Dict, List, Optional, Type

from cat.plugins.super_cat_form.super_cat_form_events import FormEvent, FormEventContext
from cat.plugins.super_cat_form.super_cat_form import SuperCatForm
from cat.plugins.super_cat_form.workflows.action_form_tool import action_tool
from cat.plugins.super_cat_form.super_cat_form import form_tool


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

    @staticmethod
    def create_sub_form_tool(form: Type[SuperCatForm], func: Callable, return_direct: bool = False, examples: Optional[List[str]] = None, *args, **kwargs) -> Any:
        """
        Create a tool for starting a sub-form.
        
        Args:
            form (Type[SuperCatForm]): The sub-form class to be used
            func (Callable): The function to be used as a tool
            return_direct (bool): Whether to return the tool directly or not
            examples (Optional[List[str]]): List of example inputs for the tool
        """

        # Create and attach the form tool
        if hasattr(form, "resolve_action_name"):
            form.resolve_action_name()
        if hasattr(form, "form_action_name") and form.form_action_name:
            return action_tool(func, return_direct=True, examples=form.start_examples, action=form.form_action_name, save_action_result=False)
        else:
            return form_tool(func, return_direct=True, examples=form.start_examples)
