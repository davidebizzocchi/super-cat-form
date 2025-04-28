from collections import defaultdict
from typing import Any, Dict, List, Literal, Set, Type, Union

from cat.plugins.super_cat_form.workflows.action import Action


class WorkflowManager:
    """
    Manage action workflow and their context.

    Features:
    - Context tracking: Manages information through key-value pairs
    - Dependency resolution: Tracks and resolves dependencies between actions
    - Execution control: Ensures actions only run when their dependencies are satisfied
    """

    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._actions: Dict[str, Action] = {}

        self._pending_actions: Dict[str, Set[str]] = defaultdict(set)  # [action, actions_that_depend_on_it]
        self._unresolved_requirements: Dict[str, int] = {}  # [action, count_of_unresolved_dependencies]


    # Context Management
    def clear_context(self):
        """Clears all stored context data."""
        self._context.clear()

    def set_context(self, key: Union[str, Action], value: Any, mode: Literal["set", "append"] = "set"):
        """Store a value in the workflow context."""
        self._execute_context_operation("add", key, value, mode)

    def update_context(self, key: Union[str, Action], value: Any, **kwargs):
        """
        Update a value in the workflow context.
        In kwargs set the search fields.
        """
        if not kwargs:
            self._execute_context_operation("add", key, value)
        else:
            self._execute_context_operation("update", key, value, **kwargs)

    def remove_context(self, key: Union[str, Action]):
        """Remove a value from the workflow context."""
        self._execute_context_operation("delete", key)

    def get_context(self, key: Union[str, Action]) -> Any:
        """Retrieve a value from the workflow context."""
        return self._execute_context_operation("get", key)

    def has_context(self, key: Union[str, Action]) -> bool:
        """Check if a key exists in the workflow context."""
        return self._execute_context_operation("has", key)

    def _execute_context_operation(self, method: Literal["add", "delete", "get", "has", "update"], key: Union[str, Action], value: Any = None, mode: Literal["set", "append"] = "set", **kwargs):
        """Execute an operation on the workflow context."""
        if isinstance(key, Action):
            key = key.name

        if method == "add":
            if mode == "set":
                self._context[key] = value

            elif mode == "append":
                if not value:
                    self._context[key] = []
                    return

                if key in self._context:
                    if not self._context[key]:
                        self._context[key] = [value]

                    if not isinstance(self._context[key], List):
                        self._context[key] = [self._context[key], value]
                    else:
                        self._context[key].append(value)
                else:
                    self._context[key] = [value]

        elif method == "update":
            if not kwargs:
                return

            if key in self._context:
                if isinstance(self._context[key], List):
                    for idx, item in enumerate(self._context[key]):
                        if isinstance(item, Dict):
                            for k, v in kwargs.items():
                                if not k in item:
                                    break

                                if item[k] != v:
                                    break
                            else:
                                if value:
                                    self._context[key][idx] = value
                                else:
                                    del self._context[key][idx]

                        # Item is not Dict
                        else:
                            for k, v in kwargs.items():
                                if not hasattr(item, k):
                                    break

                                if getattr(item, k) != v:
                                    break
                            else:
                                if value:
                                    self._context[key][idx] = value
                                else:
                                    del self._context[key][idx]

                else:
                    self._context[key] = [self._context[key], value]

        elif method == "delete":
            self._context.pop(key, None)
        elif method == "get":
            return self._context.get(key, None)
        elif method == "has":
            return key in self._context


    # Action Management
    def _get_action_id(self, name: Union[str, Action]) -> str:
        """Get the identifier for an action."""
        if isinstance(name, Action):
            return name.name

        if not isinstance(name, str):
            raise ValueError(f"Invalid action name: {name}")

        return name

    def _get_action_from_id(self, id: Union[str, Action]) -> Action:
        """Retrieve an action by its identifier."""
        if not id:
            return None

        if isinstance(id, Action):
            return id

        if not isinstance(id, str):
            raise ValueError(f"Invalid action id: {id}")

        return self._actions.get(id, None)

    def register_action(self, action: Action):
        """
        Register an action in the workflow.

        Once registered, the action's requirements will be tracked
        and it will only be marked as ready when all requirements are satisfied.
        """
        if not action:
            return None

        if action.name in self._actions:
            return

        self._actions[self._get_action_id(action)] = action

        if self._resolve_action(action):
            self._update_dependent_operations(action)

    def get_action(self, name: Union[str, Action]) -> Action:
        """Retrieve a registered action."""
        return self._get_action_from_id(name)

    def remove_action(self, action_name: Union[str, Action]):
        """
        Remove an action from the workflow.

        This will:
        1. Remove the action from the registry
        2. Update any operations that depend on it
        3. Remove its context data
        """
        action_name = self._get_action_id(action_name)

        action = self._actions.pop(action_name, None)
        if action:
            # Update pending operations
            for requirement in action.requires:
                if requirement in self._pending_actions and action_name in self._pending_actions[requirement]:
                    self._pending_actions[requirement].remove(action_name)

            # Remove from unresolved tracking
            self._unresolved_requirements.pop(action_name, -1)

            # Remove from context
            self._context.pop(action_name, None)

            # Update operations that depend on this one
            for dep in self._actions.values():
                if action_name in dep.requires:
                    self._pending_actions[action_name].add(dep.name)

    def _resolve_action(self, action: Action):
        """
        Determine if an action's requirements are satisfied.

        Returns True if all requirements are met, False otherwise.
        """
        action = self._get_action_from_id(action)

        resolved = True
        unresolved_count = 0

        # Check if all requirements are resolved
        for requirement in action.requires:
            if not self.is_action_ready(requirement, safe=False):
                # Add to pending operations
                self._pending_actions[self._get_action_id(requirement)].add(action.name)
                resolved = False
                unresolved_count += 1

        # Update unresolved count
        self._unresolved_requirements[action.name] = unresolved_count
        return resolved

    def _update_dependent_operations(self, action: Action):
        """
        Update operations that depend on this action.

        When an action is satisfied, this updates the unresolved count
        for all operations that depend on it.
        """
        action = self._get_action_from_id(action)

        # Get pending operations that depend on this one
        dependent_actions = self._pending_actions.pop(action.name, set())

        # Update each dependent operation
        for action_name in dependent_actions:
            unresolved_count = self._unresolved_requirements[action_name]

            # Decrement the unresolved count
            unresolved_count -= 1
            self._unresolved_requirements[action_name] = unresolved_count

    # Execution Control
    def execute_action(self, name: str, *args, **kwargs):
        """
        Execute an action's operation if all its requirements are met.
        """
        action = self.get_action(name)
        return action.execute(*args, **kwargs)

    def can_execute(self, name: Union[str, Action]) -> bool:
        """Check if an action can be executed."""
        return self.is_action_ready(name)

    def is_action_ready(self, name: Union[str, Action], safe: bool = True) -> bool:
        """
        Determine if an action is ready for execution.

        Args:
            name: The action to check
            safe: If True, will return True for unknown actions

        Returns:
            True if the action can be executed, False otherwise
        """
        if not name:
            return True

        if safe:
            # If action isn't registered, assume it's ready
            if not self._get_action_from_id(name):
                return True

        action_name = self._get_action_id(name)
        return self._unresolved_requirements.get(action_name, -1) == 0

    @property
    def context(self) -> Dict[str, Any]:
        """Access the current workflow context."""
        return self._context

    @property
    def actions(self) -> Dict[str, Action]:
        """Access all registered actions."""
        return self._actions
