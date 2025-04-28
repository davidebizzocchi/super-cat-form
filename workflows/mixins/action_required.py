from typing import List
from cat.experimental.form import CatFormState


class ActionRequiredMixin:
    """
    Mixin that requires actions to be completed before allowing form submission.

    This enforces workflow ordering by blocking form submission until
    required actions are satisfied.
    """

    required_actions: List = []  # List of Action objects or action names

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.required_actions:
            for action in self.required_actions:
                self.workflow.register_action(action)

        self._action_dependencies_status = CatFormState.INCOMPLETE  # Only COMPLETE or INCOMPLETE are allowed

    def check_exit_intent(self):
        """
        Block form exit if required actions aren't satisfied.
        """
        if self._action_dependencies_status == CatFormState.INCOMPLETE:
            return False
        return super().check_exit_intent()

    def confirm(self):
        """
        Block form submission if required actions aren't satisfied.
        """
        if self._action_dependencies_status == CatFormState.INCOMPLETE:
            return False
        return super().confirm()

    def validate(self):
        """
        Override validation to check action status.
        """
        self._check_required_actions()

        result = super().validate()

        if self._state == CatFormState.COMPLETE and self._action_dependencies_status == CatFormState.INCOMPLETE:
            self._state = CatFormState.INCOMPLETE

        return result

    def next(self):
        """
        Override next to check action status before proceeding.
        """
        self._check_required_actions()
        return super().next()

    def _check_required_actions(self):
        """
        Check if all required actions have completed.
        """
        if all(self.workflow.get_context(action) is not None for action in self.required_actions):
            self._action_dependencies_status = CatFormState.COMPLETE
        else:
            self._action_dependencies_status = CatFormState.INCOMPLETE
