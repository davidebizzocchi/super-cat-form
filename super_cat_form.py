import inspect
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from cat.experimental.form import CatForm, CatFormState
from cat.looking_glass.callbacks import ModelInteractionHandler, NewTokenHandler
from cat.log import log
from cat import utils
from cat.plugins.super_cat_form.super_cat_form_agent import SuperCatFormAgent
from cat.plugins.super_cat_form.super_cat_form_events import FormEventManager, FormEvent, FormEventContext
from cat.plugins.super_cat_form import prompts


def form_tool(func=None, *, return_direct: bool = False, examples: Optional[List[str]] = None):
    """
    Decorator to mark a method as a form tool.
    
    Args:
        func: The function to decorate
        return_direct: Whether to return the result directly
        examples: List of examples for the tool
        
    Returns:
        The decorated function
    """
    if examples is None:
        examples = []

    if func is None:
        return lambda f: form_tool(f, return_direct=return_direct, examples=examples)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper._is_form_tool = True
    wrapper._return_direct = return_direct
    wrapper._examples = examples
    return wrapper


class SuperCatForm(CatForm):
    """
    SuperCatForm extends CatForm with advanced functionality for handling nested forms,
    tool execution, and event management.
    """
    ner_prompt: str = prompts.DEFAULT_NER_PROMPT
    tool_prompt: str = prompts.DEFAULT_TOOL_PROMPT
    default_examples: str = prompts.DEFAULT_TOOL_EXAMPLES

    # Track the form that started this form (if any)
    parent_form: Type["SuperCatForm"] = None

    # Track the next form to activate when this form is closed
    next_form: Type["SuperCatForm"] = None

    # List of inside forms, automatically create a form_tool for calling the inside form
    inside_forms: List[Type["SuperCatForm"]] = []
    
    # Flag for cleaning up conversation history - each form is a completely new conversation
    fresh_start: bool = False

    # If True, delete all messages of this form when is closed
    delete_messages: bool = False

    # Result of (eventual) previos form
    # [form.name, form.form_data_validated]
    prev_results: Dict[str, BaseModel] = {}

    def __init__(self, cat, parent_form=None):
        """
        Initialize a new SuperCatForm instance.
        
        Args:
            cat: The cat instance
            parent_form: The parent form, if this is a sub-form
        """
        super().__init__(cat)
        
        if self.fresh_start:
            self.cat.working_memory.history = self.cat.working_memory.history[-1:]

        # Initialize inside forms
        self.initialize_inside_forms()

        # Set up form components and event handlers
        self.tool_agent = SuperCatFormAgent(self)
        self.events = FormEventManager()
        self._setup_default_handlers()
        
        # Ensure backward compatibility with version pre-1.8.0
        self._legacy_version = 'model' in inspect.signature(super().validate).parameters
        
        # Emit initialization event
        self.events.emit(
            FormEvent.FORM_INITIALIZED,
            data={},
            form_id=self.name
        )
        
        # Set custom LLM handler
        self.cat.llm = self.super_llm

        # Handle parent form relationship
        self.parent_form = parent_form or self.parent_form

        # If is set a parent_form in class definition, and is not passed in __init__
        # check if the active form is the parent
        # if yes, activate this form
        # So you can simple add parent_form attribute in class definition and manually instanziate the form
        if self.parent_form is not None and self.active_form == self.parent_form:
            self.active_form = self

        # Track the first message index for potential cleanup
        self.first_message_index = len(self.cat.working_memory.history) - 1

        # Handle next form if specified
        if self.next_form is not None:
            self.parent_form = self.next_form(
                cat=self.cat,
                parent_form=self.parent_form
            )
            self.parent_form.next()

    # -------------------------------------------------------------------------
    # LLM and Form Processing Methods
    # -------------------------------------------------------------------------

    def super_llm(self, prompt: Union[str, ChatPromptTemplate], params: Optional[Dict] = None, stream: bool = False) -> str:
        """
        Custom LLM handler for the form.
        
        Args:
            prompt: The prompt text or template
            params: Parameters for the prompt
            stream: Whether to stream the response
            
        Returns:
            The LLM response as a string
        """
        callbacks = []
        if stream:
            callbacks.append(NewTokenHandler(self.cat))

        caller = utils.get_caller_info()
        callbacks.append(ModelInteractionHandler(self.cat, caller or "StrayCat"))

        if isinstance(prompt, str):
            prompt = ChatPromptTemplate(
                messages=[
                    # Use HumanMessage instead of SystemMessage for wide-range compatibility
                    HumanMessage(content=prompt)
                ]
            )

        chain = (
                prompt
                | RunnableLambda(lambda x: utils.langchain_log_prompt(x, f"{caller} prompt"))
                | self.cat._llm
                | RunnableLambda(lambda x: utils.langchain_log_output(x, f"{caller} prompt output"))
                | StrOutputParser()
        )

        output = chain.invoke(
            params or {},
            config=RunnableConfig(callbacks=callbacks)
        )

        return output

    def update(self):
        """
        Update the form model with extracted data, maintaining compatibility with
        both old and new CatForm versions.
        """
        old_model = self._model.copy() if self._model is not None else {}

        # Extract and sanitize new data
        json_details = self.extract()
        json_details = self.sanitize(json_details)
        merged_model = old_model | json_details

        if self._legacy_version:
            # old version: validate returns the updated model
            validated_model = self.validate(merged_model)
            # ensure we never set None as the model
            self._model = validated_model if validated_model is not None else {}
        else:
            # new version: set model first, then validate
            self._model = merged_model
            self.validate()

        # ensure self._model is never None
        if self._model is None:
            self._model = {}

        # emit events for updated fields
        updated_fields = {
            k: v for k, v in self._model.items()
            if k not in old_model or old_model[k] != v
        }

        if updated_fields:
            self.events.emit(
                FormEvent.FIELD_UPDATED,
                {
                    "fields": updated_fields,
                    "old_values": {k: old_model.get(k) for k in updated_fields}
                },
                self.name
            )

    def extract(self):
        """
        Extract form data from chat history using NER with LangChain JsonOutputParser.
        
        Returns:
            Dict: Extracted form data
        """
        try:
            self.events.emit(
                FormEvent.EXTRACTION_STARTED,
                data={
                    "chat_history": self.cat.stringify_chat_history(),
                    "form_data": self.form_data
                },
                form_id=self.name
            )
            prompt_params = {
                "chat_history": self.cat.stringify_chat_history(),
                "form_description": f"{self.name} - {self.description}"
            }
            parser = JsonOutputParser(pydantic_object=self.model_getter())
            prompt = PromptTemplate(
                template=self.ner_prompt,
                input_variables=list(prompt_params.keys()),
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            chain = prompt | self.cat._llm | parser
            output_model = chain.invoke(prompt_params)
            self.events.emit(
                FormEvent.EXTRACTION_COMPLETED,
                data=output_model,
                form_id=self.name
            )
        except Exception as e:
            output_model = {}
            log.error(e)

        return output_model

    def sanitize(self, model: Dict) -> Dict:
        """
        Sanitize the model while preserving nested structures.
        Only removes explicitly null values.

        Args:
            model: Dictionary containing form data

        Returns:
            Dict: Sanitized form data
        """
        if "$defs" in model:
            del model["$defs"]

        def _sanitize_nested(data):
            if isinstance(data, dict):
                return {
                    k: _sanitize_nested(v)
                    for k, v in data.items()
                    if v not in ("None", "null", "lower-case", "unknown", "missing")
                }
            return data

        return _sanitize_nested(model)

    def validate(self, model=None):
        """
        Validate the form model against the Pydantic model class.
        
        Args:
            model: Optional model data to validate instead of self._model
            
        Returns:
            Validated model data for legacy compatibility
        """
        self.events.emit(
            FormEvent.VALIDATION_STARTED,
            {"model": self._model},
            self.name
        )

        self._missing_fields = []
        self._errors = []

        try:
            if self._legacy_version and model is not None:
                validated_model = self.model_getter()(**model).model_dump(mode="json")
                self._state = CatFormState.COMPLETE
                return validated_model
            else:
                # New version: validate self._model
                self.model_getter()(**self._model)
                self._state = CatFormState.COMPLETE

        except ValidationError as e:
            for error in e.errors():
                field_path = '.'.join(str(loc) for loc in error['loc'])
                if error['type'] == 'missing':
                    self._missing_fields.append(field_path)
                else:
                    self._errors.append(f'{field_path}: {error["msg"]}')

            self._state = CatFormState.INCOMPLETE

            if self._legacy_version and model is not None:
                return model
        finally:
            self.events.emit(
                FormEvent.VALIDATION_COMPLETED,
                {
                    "model": self._model,
                    "missing_fields": self._missing_fields,
                    "errors": self._errors
                },
                self.name
            )

    def next(self):
        """
        Process the next step in the form.
        
        Returns:
            Dict with output content for the UI
        """
        if self._state == CatFormState.WAIT_CONFIRM:
            if self.confirm():
                self._state = CatFormState.CLOSED
                self.events.emit(
                    FormEvent.FORM_SUBMITTED,
                    {
                        "form_data": self.form_data
                    },
                    self.name
                )
                return self.submit_close(self._model)
            else:
                if self.check_exit_intent():
                    self._state = CatFormState.CLOSED
                    self.events.emit(
                        FormEvent.FORM_CLOSED,
                        {
                            "form_data": self.form_data
                        },
                        self.name
                    )
                else:
                    self._state = CatFormState.INCOMPLETE

        if self.check_exit_intent() and self._state != CatFormState.CLOSED:
            self._state = CatFormState.CLOSED
            self.events.emit(
                FormEvent.FORM_CLOSED,
                {
                    "form_data": self.form_data
                },
                self.name
            )

        if self._state == CatFormState.INCOMPLETE:
            # Execute agent if form tools are present
            if len(self.get_form_tools()) > 0:
                agent_output = self.tool_agent.execute(self.cat)
                if agent_output.output:
                    if agent_output.return_direct:
                        return {"output": agent_output.output}
                self.update()
            else:
                self.update()

        if self._state == CatFormState.COMPLETE:
            if self.ask_confirm:
                self._state = CatFormState.WAIT_CONFIRM
            else:
                self._state = CatFormState.CLOSED
                return self.submit_close(self._model)

        return self.message()

    def model_getter(self) -> Type[BaseModel]:
        """
        Get the Pydantic model class for the form.
        Override for backward compatibility with older CatForm versions.
        
        Returns:
            The Pydantic model class
        """
        return self.model_class

    # -------------------------------------------------------------------------
    # Form Tool and Inside Form Management
    # -------------------------------------------------------------------------

    @classmethod
    def get_form_tools(cls):
        """
        Get all methods of the class that are decorated with @form_tool.
        
        Returns:
            Dict of form tool methods
        """
        form_tools = {}
        for name, func in inspect.getmembers(cls):
            if inspect.isfunction(func) or inspect.ismethod(func):
                if getattr(func, '_is_form_tool', False):
                    form_tools[name] = func
        return form_tools

    @staticmethod
    def format_class_name(name):
        """
        Formats a class name into snake_case by inserting underscores before capital letters
        and converting the result to lowercase.

        Args:
            name (str): The class name to format.

        Returns:
            str: The formatted name in snake_case.
        """
        def replacement(match):
            """Helper function to determine the replacement for regex matches."""
            # If the match is from the second pattern (?<!^)(?=[A-Z]), insert an underscore
            if match.group(0) == '':
                return '_'
            # If the match is from the first pattern ([A-Z]+)([A-Z][a-z]), insert underscore between groups
            return match.group(1) + '_' + match.group(2)

        # Regex pattern to handle
        pattern = r'([A-Z]+)([A-Z][a-z])|(?<!^)(?=[A-Z])'

        # Apply the regex substitution, convert to lowercase, and replace spaces with underscores
        return re.sub(pattern, replacement, name).lower().replace(" ", "_")

    @classmethod
    def initialize_inside_forms(cls):
        """
        Initializes inside forms for the current form by creating dynamic form_tool methods.
        """
        for form_class in cls.inside_forms:
            if issubclass(form_class, CatForm):
                # Format the form class name into snake_case
                formatted_form_name = cls.format_class_name(form_class.name or form_class.__name__)
                tool_name = f"start_form_{formatted_form_name}"

                # Define a dynamic method to start the inside form
                def tool_start_inside_form(self, *args):
                    return self.start_sub_form(form_class)

                # Set the docstring of the dynamic method to include the example
                # All examples are joined with " or " in the tool docstring
                tool_start_inside_form.__doc__ = " or ".join(form_class.start_examples) + ". Input is always None."

                # Set the name of the dynamic form method
                tool_start_inside_form.__name__ = tool_name

                # Wrap the dynamic method as a form tool
                wrapped_form_tool = form_tool(
                    func=tool_start_inside_form,
                    return_direct=True
                )

                # Set the methods to the class
                setattr(cls, tool_name, wrapped_form_tool)

                log.debug(f"Add inside form {form_class.name} with docstring: {tool_start_inside_form.__doc__}")

    def start_sub_form(self, form_class):
        """
        Create and activate a new sub-form, saving this form as the parent.
        
        Args:
            form_class: The form class to instantiate
            
        Returns:
            str: The initial message from the new form
        """
        # Create the new form instance
        new_form = form_class(
            self.cat,
            parent_form=self
        )

        # Activate the new form
        self.cat.working_memory.active_form = new_form

        # Emit event for the new form activation 
        self.events.emit(
            FormEvent.INSIDE_FORM_ACTIVE,
            {
                "instance": new_form
            },
            self.name
        )

        log.debug(f"Started sub-form: {new_form.name} from parent: {self.name}")

        # Return the first message of the new form
        return new_form.next()["output"]
    
    def create_prev_results_in_parent(self, *args, **kwargs):
        """
        Create the prev_results in the parent form with this form's validated data.
        """
        self.parent_form.prev_results = self.prev_results.copy()
        self.parent_form.prev_results[self.name] = self.form_data_validated

    def submit_close(self, form_data):
        """
        Submit the form and handle parent form relationships.
        
        Args:
            form_data: The form data to submit
            
        Returns:
            AgentOutput: The message to display after submission
        """
        if self.parent_form is not None:
            self.create_prev_results_in_parent()

            self.parent_form.events.emit(
                FormEvent.INSIDE_FORM_CLOSED,
                {
                    "form_data": form_data,
                    "output": self.submit(form_data)
                },
                self.name
            )

            # Return message of the external (old) form
            return self.parent_form.message()

        # By default, return the submit output
        return self.submit(form_data)

    def message_closed(self, force=False):
        """
        Return the message of previous form if exists, otherwise the default message.
        
        Args:
            force: Whether to force the default closed message
            
        Returns:
            AgentOutput: The message to display
        """
        if self.parent_form is not None and not force:
            return self.parent_form.message()

        return super().message_closed()

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def _setup_default_handlers(self):
        """Setup default event handlers for the form."""
        for event in FormEvent:
            self.events.on(event, self._log_event)

        # Setup event handlers for various form events
        self.events.on(FormEvent.INSIDE_FORM_ACTIVE, self._on_inside_create_form)
        self.events.on(FormEvent.INSIDE_FORM_CLOSED, self._on_inside_form_closed)
        self.events.on(FormEvent.FORM_CLOSED, self._on_form_closed)
        
        # Reset the active form when the form is submitted or closed
        self.events.on(FormEvent.FORM_SUBMITTED, self._restore_parent_form)
        self.events.on(FormEvent.FORM_CLOSED, self._restore_parent_form)
        
        # Next form activation and previous form inactivation
        self.events.on(FormEvent.NEXT_FORM_ACTIVE, self._on_next_form_actived)
        self.events.on(FormEvent.PREVIOUS_FORM_INACTIVE, self._on_previous_form_inactive)

    def _log_event(self, event: FormEventContext):
        """Log form events for debugging."""
        log.debug(f"Form {self.name}: {event.event.name} - {event.data}")

    def _on_inside_create_form(self, context: FormEventContext):
        """
        Called when a new inside form is created.
        
        Args:
            context: Event context with data about the created form
        """
        log.debug(f"[EVENT: _on_inside_create_form] inside form in {self.name} created")
        form_class = context.data.get("instance")
        log.debug(f"Creating inside form: {form_class}")

    def _on_inside_form_closed(self, context: FormEventContext):
        """
        Called when an inside form is closed.
        
        Args:
            context: Event context with form data and output
        """
        submit_output = context.data.get("output")
        form_data = context.data.get("form_data")

        # Send the submit output to chat
        self.cat.send_chat_message(submit_output["output"])

    def _on_form_closed(self, context: FormEventContext):
        """
        Called when the form is closed.
        
        Args:
            context: Event context
        """
        log.debug(f"[EVENT: _on_form_closed] form {self.name} closed")

        if self.parent_form is not None:
            self.parent_form.events.emit(
                FormEvent.INSIDE_FORM_CLOSED,
                {
                    "form_data": context.data.get("form_data"),
                    "output": self.message_closed(force=True)
                },
                self.name
            )

    def _on_next_form_actived(self, context: FormEventContext):
        """
        Called when a next form is activated.
        
        Args:
            context: Event context
        """
        log.debug(f"[EVENT: _on_next_form_actived] {self.name} new next form activated")

    def _on_previous_form_inactive(self, context: FormEventContext):
        """
        Called when the previous form is inactivated.
        
        Args:
            context: Event context
        """
        log.debug(f"[EVENT: _on_previous_form_inactive] {self.name} previous form is inactive (by next_form)")

    def _restore_parent_form(self, *args, **kwargs):
        """
        Reset the active form to the previous form, if exists.
        """
        if self.parent_form is not None:
            self.active_form = self.parent_form

            if self.next_form is not None:
                self.events.emit(
                    FormEvent.NEXT_FORM_ACTIVE,
                    {
                        "instance": self.parent_form
                    },
                    self.name
                )

                self.parent_form.events.emit(
                    FormEvent.PREVIOUS_FORM_INACTIVE,
                    {},
                    self.name
                )

        if self.delete_messages:
            del self.cat.working_memory.history[self.first_message_index:]

    def _search_indexs_of_messages(self):
        """
        Find the start and end indexes of messages related to this form.
        
        Returns:
            Tuple[int, int]: Start and end indexes
        """
        start_idx = -1
        end_idx = -1
        is_first = True

        for i in range(len(self.cat.working_memory.history)):
            message = self.cat.working_memory.history[i]

            # Interested only in AI messages
            if message.who == "AI":
                for step in message.why.intermediate_steps:
                    # Check if the message is send from this form
                    if step[0][0] == self.name:
                        # Check all previous user messages
                        if is_first:
                            is_first = False

                            if i == 0:
                                break

                            # Search backward until a non-"Human" message is found
                            for j in range(i-1, -1, -1):
                                j_message = self.cat.working_memory.history[j]

                                if j_message.who == "Human":
                                    start_idx = j
                                # If is not "Human", stop search
                                else:
                                    break

                        # Update the last index
                        end_idx = i
                        break

        return start_idx, end_idx

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def form_data(self) -> Dict:
        """Get the current form data."""
        return self._model

    @property
    def form_data_validated(self) -> Optional[BaseModel]:
        """Get the validated form data as a Pydantic model."""
        return self._get_validated_form_data()

    def _get_validated_form_data(self) -> Optional[BaseModel]:
        """
        Safely attempts to get validated form data.
        
        Returns:
            Optional[BaseModel]: Validated Pydantic model if successful, None otherwise
        """
        try:
            return self.model_getter()(**self._model)
        except ValidationError:
            return None

    @property
    def active_form(self):
        """Get the currently active form."""
        return self.cat.working_memory.active_form

    @active_form.setter
    def active_form(self, form):
        """Set the currently active form."""
        self.cat.working_memory.active_form = form


def super_cat_form(form: Type[SuperCatForm]) -> Type[SuperCatForm]:
    """
    Decorator to mark a class as a SuperCatForm.
    
    Args:
        form: The form class to decorate
        
    Returns:
        The decorated form class
    """
    form._autopilot = True
    if form.name is None:
        form.name = form.__name__

    if form.triggers_map is None:
        form.triggers_map = {
            "start_example": form.start_examples,
            "description": [f"{form.name}: {form.description}"],
        }

    return form