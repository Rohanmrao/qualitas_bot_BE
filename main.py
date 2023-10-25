import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextCompletion
from semantic_kernel import (
    ChatPromptTemplate,
    SemanticFunctionConfig,
    PromptTemplateConfig,
)

import time
import re   # for splitting the response into sections

from flask import Flask, request, jsonify
from flask_cors import CORS # for cross origin requests since the frontend is on a react app with port 3000

app = Flask(__name__)
CORS(app, origins="https://qualitas-bot-fe.vercel.app/")

# Static variables and directories
import json
CONFIG_DIRECTORY = "chat_config.json"

chat_config_dict = json.load(open(CONFIG_DIRECTORY))
# print(chat_config_dict, type(chat_config_dict))

# Kernel initialization
kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_chat_service(
    "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))


def print_ai_services(kernel):
    print(
        f"Text completion services added: {kernel.all_text_completion_services()}")
    print(f"Chat completion services added: {kernel.all_chat_services()}")
    print(
        f"Text embedding generation services added: {kernel.all_text_embedding_generation_services()}"
    )


# list of all the services added to the kernel
print_ai_services(kernel)


def create_semantic_function_config(prompt_template, prompt_config_dict, kernel):
    chat_system_message = (
        prompt_config_dict.pop("system_prompt")
        if "system_prompt" in prompt_config_dict
        else None
    )

    # understand the context and other hyperparams here
    prompt_template_config = PromptTemplateConfig.from_dict(prompt_config_dict)
    prompt_template_config.completion.token_selection_biases = (
        {}
    )

    # the config and template are combined to create the prompt_template object
    prompt_template = ChatPromptTemplate(
        template=prompt_template,
        prompt_config=prompt_template_config,
        template_engine=kernel.prompt_template_engine,
    )

    if chat_system_message is not None:
        prompt_template.add_system_message(chat_system_message)

    #returning the entire semantic function object
    return SemanticFunctionConfig(prompt_template_config, prompt_template)

#register the newly created cutome semantic function using the built in register_semantic_function method
chatbot = kernel.register_semantic_function(
    skill_name="Chatbot",
    function_name="chatbot",
    function_config=create_semantic_function_config(
        "{{$input}}", chat_config_dict, kernel
    )
)

# This is a function that displays the repsonse word by word with a time delay set (to mimic the typing of a human). Sentences are broken according to serial numbers
def display_word_by_word(text, delay=0.1):
    sections = re.split(r"\d+\.", str(text))
    for section in sections:
        words = section.split()
        for word in words:
            print(word, end=' ', flush=True)
            time.sleep(delay)
        print("\n")

@app.route('/', methods=['GET'])
def push_response():
    
    user_input = str(request.args.get('user_input'))
    print(user_input)
    ai_response = chatbot(user_input)
    # print(ai_response)
    return str(ai_response)
    

if __name__ == "__main__":
    app.run()
