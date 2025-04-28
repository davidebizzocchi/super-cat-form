from typing import Dict, Any

from cat.plugins.super_cat_form.super_cat_form_agent import SuperCatFormAgent
from cat.plugins.super_cat_form.workflows.manager import WorkflowManager


class WorkflowAgent(SuperCatFormAgent):
    """
    Workflow-aware agent that filters tools based on dependency relationships.

    This agent extends SuperCatFormAgent by:
    1. Integrating with the workflow dependency system
    2. Filtering tools based on their readiness in the workflow
    3. Presenting only available operations to the user
    """

    def __init__(self, form_instance, workflow: WorkflowManager):
        """
        Initialize the workflow-aware agent.

        Args:
            form_instance: The form instance this agent belongs to
            workflow: The workflow manager for tracking dependencies
        """
        super().__init__(form_instance)
        self.workflow = workflow
        self._form_tools = form_instance.get_form_tools()

    @property
    def form_tools(self) -> Dict[str, Any]:
        """
        Override form_tools to return only tools that are available in the workflow.

        A tool is available if:
        - It has no action specified (_action is None)
        - Its action can be executed (all requirements are satisfied)

        Returns:
            Dict of available form tools filtered by dependency status
        """
        return {
            name: tool
            for name, tool in self._form_tools.items()
            if self._can_execute_tool(tool)
        }

    @form_tools.setter
    def form_tools(self, value: Dict[str, Any]):
        """
        Setter for form_tools.
        Set the "base" form tools.
        """
        self._form_tools = value

    def _can_execute_tool(self, tool: Any) -> bool:
        """
        Determine if a tool can be executed based on its action attribute.

        Args:
            tool: The tool to check

        Returns:
            True if the tool can be executed, False otherwise
        """
        action = getattr(tool, "_action", None)
        if action is None:
            return True
        return self.workflow.can_execute(action)
