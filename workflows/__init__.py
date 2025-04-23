"""
Workflow module for SuperCatForm that provides dependency management for multi-step processes.

This module introduces a workflow system that tracks dependencies between operations
and manages context data to ensure operations are executed in the correct order.
"""


# Core components
from cat.plugins.super_cat_form.workflows.action import Action
from cat.plugins.super_cat_form.workflows.manager import WorkflowManager
from cat.plugins.super_cat_form.workflows.agent import WorkflowAgent
from cat.plugins.super_cat_form.workflows.form import WorkflowForm

from cat.plugins.super_cat_form.workflows.mixins import (
    FormActionMixin,
    ContextWorkflowMixin,
    ActionRequiredMixin
)

__all__ = [
    # Core components
    "Action",
    "WorkflowManager",
    "WorkflowAgent",
    "WorkflowForm",
    "FormActionMixin",
    "ContextWorkflowMixin",
    "ActionRequiredMixin"
]