from functools import wraps
from cat.plugins.super_cat_form.super_cat_form import form_tool
from cat.log import log


def action_tool(func=None, *, return_direct=False, examples=None, action=None, save_action_result=True):

    if examples is None:
        examples = []

    if func is None:
        return lambda f: action_tool(f, return_direct=return_direct, examples=examples, action=action, save_action_result=save_action_result)


    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        if action and save_action_result:
            self.workflow.set_context(action, result)
        
        return result


    wrapper._is_form_tool = True
    wrapper._return_direct = return_direct
    wrapper._examples = examples
    wrapper._action = action
    return wrapper
