from cat.plugins.super_cat_form.super_cat_form_events import FormEvent, FormEventContext


class FormActionMixin:
    """
    Mixin that registers a form as an action in the workflow.
    
    This allows forms to become action nodes that other workflow
    components can depend on.
    """
    form_action = None  # Action object
    action_name: str = None

    def __init__(self, *args, **kwargs):
        from cat.plugins.super_cat_form.workflows.action import Action

        super().__init__(*args, **kwargs)

        if self.form_action:
            self.action_name = self.form_action.name
        elif not self.action_name:
            self.action_name = self.name

        if not self.form_action:
            self.form_action = self.workflow.get_action(self.action_name)

        if not self.form_action:
            self.form_action = Action(name=self.action_name, operation=self)
            self.workflow.register_action(self.form_action)

    def _setup_default_handlers(self):
        super()._setup_default_handlers()

        self.events.on(FormEvent.FORM_SUBMITTED, self.save_form_action)

    def save_form_action(self, context: FormEventContext):
        """
        Save the form data as an action result in the workflow context.
        """
        self.workflow.set_context(self.action_name, self.form_data_validated)

    @classmethod
    def resolve_action_name(cls):
        """
        Resolve the action name from the form_action.
        """
        if cls.form_action:
            cls.action_name = cls.form_action.name
            return True
        return False
