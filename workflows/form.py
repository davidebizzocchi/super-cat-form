from cat.plugins.super_cat_form.workflows.manager import WorkflowManager
from cat.plugins.super_cat_form.workflows.agent import WorkflowAgent
from cat.plugins.super_cat_form.super_cat_form import SuperCatForm


class WorkflowForm(SuperCatForm):
    """
    Form with integrated workflow management for orchestrating multi-step processes.
    
    This form type:
    1. Uses a workflow manager to track dependencies between operations
    2. Provides a specialized agent that respects the workflow sequence
    3. Enables building complex multi-form workflows with dependency chains
    """

    def __init__(self, cat, workflow: WorkflowManager = None):
        """
        Initialize a workflow-enabled form.
        
        Args:
            cat: The CheshireCat instance
            workflow: Optional workflow manager. If not provided, a new one is created.
        """
        super().__init__(cat)
        self.workflow = workflow or WorkflowManager()
        self.tool_agent = WorkflowAgent(self, self.workflow)
