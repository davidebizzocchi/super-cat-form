DEFAULT_NER_PROMPT = """
You are an advanced AI assistant specializing in information extraction and structured data formatting. 
Your task is to extract relevant information from a given text and format it according to a specified JSON structure. 
This extraction is part of a conversational form-filling process.

Here are the key components for this task:

1. Chat History:
<chat_history>
{chat_history}
</chat_history>

2.Form Description:
<form_description>
{form_description}
</form_description>

3. Format Instructions (JSON schema):
<format_instructions>
{format_instructions}
</format_instructions>

Remember:
- The extraction is part of an ongoing conversation, so consider any contextual information that might be relevant.
- Only include information that is explicitly stated or can be directly inferred from the input text.
- If a required field in the JSON schema cannot be filled based on the available information, use null or an appropriate default value as specified in the format instructions.
- Ensure that the output JSON is valid and matches the structure defined in the format instructions.

"""

DEFAULT_TOOL_PROMPT = """
Create a JSON with the correct "action" and "action_input" for form compilation assistance.

Current form data: {form_data}

Available actions: {tools}

CORE RULES:
1. Use specific tools ONLY when explicitly requested by user
2. Use "form_completion" for:
   - Any form filling or ordering intention
   - Direct responses to form questions
3. Default to "no_action" for:
   - When no action needed

{examples}

Response Format:
{{
    "action": str,  // One of [{tool_names}, "no_action", "form_completion"]
    "action_input": str | null  // Per action description
}}
"""

DEFAULT_TOOL_EXAMPLES = """
Examples:
"What's on the menu?" → "get_menu" (explicit menu request)
"I want to order pizza" → "form_completion" (ordering intention)
"Hi there" → "no_action" (greeting)
"""

DEFAULT_HUMAN_READABLE_PROMPT = """
You are an advanced AI assistant that specializes in formatting and make more human-readble the JSON data.
Your task is to take the structured JSON data and convert it into a more human-readable format.
The scope of the user is insert all the data the application needs.

Data are structured using JSON format, but the user needs a more human-readable version of the data to understand it better.
Are included also missing fields and invalid fields that user insert, so also help the user to understand what is wrong with the data and correctly insert it.

NON HUMAN READABLE MESSAGE:
{message}


Your message is the message read, so not explain what you are doing, but only the result of your work.
"""