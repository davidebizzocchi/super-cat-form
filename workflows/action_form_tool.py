from plugins.super_cat_form.super_cat_form import form_tool


def action_tool(func=None, *, return_direct=False, examples=None, action=None):
    """
    Decorator to add an action to a form tool.
    """

    wrapper = form_tool(func, return_direct=return_direct, examples=examples, action=action)

    wrapper._action = action

    return wrapper
