"""
Mixins for extending WorkflowForm functionality with specialized capabilities.

Available mixins:
- FormActionMixin: Registers a form as an action in the workflow
- ContextWorkflowMixin: Manages sub-forms and their workflow context
- ActionRequiredMixin: Requires actions to be satisfied before form submission
"""


from cat.plugins.super_cat_form.workflows.mixins.form_action import FormActionMixin
from cat.plugins.super_cat_form.workflows.mixins.context_workflow import ContextWorkflowMixin
from cat.plugins.super_cat_form.workflows.mixins.action_required import ActionRequiredMixin


__all__ = [
    "FormActionMixin",
    "ContextWorkflowMixin",
    "ActionRequiredMixin"
]