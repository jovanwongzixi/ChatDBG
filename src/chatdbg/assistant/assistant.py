import json
import collections
import string
import textwrap
import time
import pprint
import os

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import litellm
from openai import OpenAI, OpenAIError

from ..util.trim import sandwich_tokens, trim_messages
from ..util.text import strip_ansi
from .listeners import Printer

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")
MODEL = os.getenv("OPENAI_MODEL")

class AssistantError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def _merge_chunks(chunks):
    # Check for a final usage chunk, and merge it with the last chunk.
    if not chunks[-1].choices and chunks[-1].usage:
        chunks[-2].usage = chunks[-1].usage
        chunks.pop()

    assert all(len(chunk.choices) == 1 for chunk in chunks)

    finish_reason = chunks[-1].choices[0].finish_reason
    usage = chunks[-1].usage
    content = "".join(
        chunk.choices[0].delta.content
        for chunk in chunks
        if chunk.choices[0].delta.content  # It can be None for tool calls.
    )

    tool_chunks = [
        bit
        for chunk in chunks
        if chunk.choices[0].delta.tool_calls
        for bit in chunk.choices[0].delta.tool_calls
    ]
    tool_calls = collections.defaultdict(
        lambda: {"id": "", "name": "", "arguments": ""}
    )
    for tool_chunk in tool_chunks:
        if tool_chunk.id:
            tool_calls[tool_chunk.index]["id"] += tool_chunk.id
        if tool_chunk.function.name:
            tool_calls[tool_chunk.index]["name"] += tool_chunk.function.name
        if tool_chunk.function.arguments:
            tool_calls[tool_chunk.index]["arguments"] += tool_chunk.function.arguments

    tool_calls = [
        {
            "id": tool_call["id"],
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
            "type": "function",
        }
        for tool_call in tool_calls.values()
    ]

    return finish_reason, content, tool_calls, usage
def remove_non_printable_chars(s: str) -> str:
    printable_chars = set(string.printable)
    filtered_string = "".join(filter(lambda x: x in printable_chars, s))
    return filtered_string


class Assistant:
    def __init__(
        self,
        instructions,
        model=MODEL,
        timeout=30,
        listeners=[Printer()],
        functions=[],
        max_call_response_tokens=2048,
    ):

        # Hide their debugging info -- it messes with our error handling
        litellm.suppress_debug_info = True

        self._clients = listeners

        self._functions = {}
        for f in functions:
            self._add_function(f)

        self._model = MODEL # overwrite gpt-4o initialisation in config for now
        self._openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        self._timeout = timeout
        self._conversation = [{"role": "system", "content": instructions}]
        self._max_call_response_tokens = max_call_response_tokens

        # self._check_model()
        self._broadcast("on_begin_dialog", instructions)

    def close(self):
        self._broadcast("on_end_dialog")

    def _warn_about_exception(self, e, message="Unexpected Exception"):
        import traceback

        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        tb_string = "".join(tb_lines)
        self._broadcast("on_error", f"{message}\n\n{e}\n{tb_string}")

    def query(self, prompt: str, user_text):
        """
        Send a query to the LLM.
          - prompt is the prompt to send.
          - user_text is what the user typed (which may or not be the same as prompt)

        Returns a dictionary containing:
            - "completed":          True of the query ran to completion.
            - "cost":               Cost of query, or 0 if not completed.
        Other fields only if completed is True
            - "time":               completion time in seconds
            - "model":              the model used.
            - "tokens":             total tokens
            - "prompt_tokens":      our prompts
            - "completion_tokens":  the LLM completions part
        """
        result = {"completed": False, "cost": 0}
        start = time.time()

        self._broadcast("on_begin_query", prompt, user_text)
        try:
            self._streamed_query(prompt, user_text)
            elapsed = time.time() - start

        #     results["time"] = elapsed
        #     results["model"] = self._model
        #     results["completed"] = True
        #     results["message"] = f"\n[Cost: ~${results['cost']:.2f} USD]"
        # except OpenAIError as e:
        #     self._warn_about_exception(e, f"Unexpected OpenAI Error.  Retry the query.")
        #     results["message"] = f"[Exception: {e}]"
            # litellm model data not used for us
            # if self._model in litellm.model_data:
            #     model_data = litellm.model_data[self._model]
            #     result["cost"] = (
            #         stats["prompt_tokens"] * model_data["input_cost_per_token"]
            #         + stats["completion_tokens"] * model_data["output_cost_per_token"]
            #     )
            #     result["message"] = f"\n[Cost: ~${result['cost']:.2f} USD]"

            result["time"] = elapsed
            result["model"] = self._model
            result["completed"] = True
        except KeyboardInterrupt:
            # user action -- just ignore
            result["message"] = "[Chat Interrupted]"
        except Exception as e:
            self._warn_about_exception(e, f"Unexpected Exception.")
            result["message"] = f"[Exception: {e}]"

        self._broadcast("on_end_query", result)
        return result

    def _report(self, stats):
        if stats["completed"]:
            print()
        else:
            print("[Chat Interrupted]")

    def _broadcast(self, method_name, *args):
        for client in self._clients:
            method = getattr(client, method_name, None)
            if callable(method):
                method(*args)

    def _check_model(self):
        # result = litellm.validate_environment(self._model)
        # missing_keys = result["missing_keys"]
        # if missing_keys != []:
        #     _, provider, _, _ = litellm.get_llm_provider(self._model)
        #     if provider == "openai":
        #         raise AssistantError(
        #             textwrap.dedent(
        #                 f"""\
        #             You need an OpenAI key to use the {self._model} model.
        #             You can get a key here: https://platform.openai.com/api-keys.
        #             Set the environment variable OPENAI_API_KEY to your key value."""
        #             )
        #         )
        #     else:
        #         raise AssistantError(
        #             textwrap.dedent(
        #                 f"""\
        #             You need to set the following environment variables
        #             to use the {self._model} model: {', '.join(missing_keys)}."""
        #             )
        #         )

        # try:
        #     if not litellm.supports_function_calling(self._model):
        #         raise AssistantError(
        #             textwrap.dedent(
        #                 f"""\
        #             The {self._model} model does not support function calls.
        #             You must use a model that does, eg. gpt-4."""
        #             )
        #         )
        # except:
        #     raise AssistantError(
        #         textwrap.dedent(
        #             f"""\
        #         {self._model} does not appear to be a supported model.
        #         See https://docs.litellm.ai/docs/providers."""
        #         )
        #     )
        try:
            result = self._openai_client.models.retrieve(self._model)
            print(result)
        except OpenAIError:
            raise AssistantError(
                textwrap.dedent(
                    f"""\
                {self._model} does not appear to be a supported model. {result}
                """
                )
            )


    def _add_function(self, function):
        """
        Add a new function to the list of function tools.
        The function should have the necessary json spec as its docstring
        """
        schema = json.loads(function.__doc__)
        assert "name" in schema, "Bad JSON in docstring for function tool."
        self._functions[schema["name"]] = {"function": function, "schema": schema}

    def _make_call(self, tool_call) -> str:
        name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
            function = self._functions[name]
            call, result = function["function"](**args)
            result = remove_non_printable_chars(strip_ansi(result).expandtabs())
            self._broadcast("on_function_call", call, result)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            # likely to be an exception from the code we ran, not a bug...
            result = f"Exception in function call: {e}"
            self._broadcast("on_warn", result)
        return result

    def _streamed_query(self, prompt: str, user_text):
        cost = 0

        self._conversation.append({"role": "user", "content": prompt})
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while True: # break only when finish_reason == "stop"
            stream = self._stream_completion()

            # litellm.stream_chunk_builder is broken for new GPT models
            # that have content before calls, so...

            # stream the response, collecting the tool_call parts separately
            # from the content
            try:
                self._broadcast("on_begin_stream")
                chunks = []
                # tool_chunks = []
                for chunk in stream:
                    # print(chunk)
                    chunks.append(chunk)
                    if chunk.choices:
                        assert len(chunk.choices) == 1
                        if chunk.choices[0].delta.content != None:
                            self._broadcast(
                                "on_stream_delta", chunk.choices[0].delta.content
                            )
                        # else:
                        #     tool_chunks.append(chunk)
            finally:
                self._broadcast("on_end_stream")
                print(chunks[-1])
                
            
            # print(f"Chunks: {chunks}")
            # print(f"messages: {self._conversation}")
            # then compute for the part that litellm gives back.
            # completion = litellm.stream_chunk_builder(
            #     chunks, messages=self._conversation
            # )
            # # raise Exception("test")
            # cost += litellm.completion_cost(completion)

            # # add content to conversation, but if there is no content, then the message
            # # has only tool calls, and skip this step
            # response_message = completion.choices[0].message
            # if response_message.content != None:
            #     # fix: remove tool calls.  They get added below.
            #     response_message = response_message.copy()
            #     response_message["tool_calls"] = None
            #     self._conversation.append(response_message.json())

            # if response_message.content != None:
            #     self._broadcast("on_response", response_message.content)

            # if completion.choices[0].finish_reason == "tool_calls":
            #     # create a message with just the tool calls, append that to the conversation, and generate the responses.
            #     tool_completion = litellm.stream_chunk_builder(
            #         tool_chunks, self._conversation
            #     )

            #     # this part wasn't counted above...
            #     cost += litellm.completion_cost(tool_completion)
            finish_reason, content, tool_calls, usage_delta = _merge_chunks(chunks)
            # usage["prompt_tokens"] += usage_delta.prompt_tokens
            # usage["completion_tokens"] += usage_delta.completion_tokens
            # usage["total_tokens"] += usage_delta.total_tokens
            if content:
                self._conversation.append({"role": "assistant", "content": content})
                self._broadcast("on_response", content)
                continue
            if finish_reason == "tool_calls":
                self._conversation.append(
                    {"role": "assistant", "tool_calls": tool_calls}
                )
                self._add_function_results_to_conversation(tool_calls)
                continue
            # if finish_reason == "stop":
            #     break
            break
                # tool_message = tool_completion.choices[0].message

                # tool_json = tool_message.json()

                # patch for litellm sometimes putting index fields in the tool calls it constructs
                # in stream_chunk_builder.  gpt-4-turbo-2024-04-09 can't handle those index fields, so
                # just remove them for the moment.
            #     for tool_call in tool_json.get("tool_calls", []):
            #         _ = tool_call.pop("index", None)

            #     tool_json["role"] = "assistant"
            #     self._conversation.append(tool_json)
            #     self._add_function_results_to_conversation(tool_message)
            # else:
                # break

        # results = {
        #     "cost": cost,
        #     "tokens": completion.usage.total_tokens,
        #     "prompt_tokens": completion.usage.prompt_tokens,
        #     "completion_tokens": completion.usage.completion_tokens,
        # }
        # return usage

    def _stream_completion(self):

        self._trim_conversation()
        # print(self._functions.values())
        return self._openai_client.chat.completions.create(
            model=self._model,
            messages=self._conversation,
            tools=[
                {"type": "function", "function": f["schema"]}
                for f in self._functions.values()
            ],
            timeout=self._timeout,
            stream=True,
            # stream_options={"include_usage": True},
        )

    def _trim_conversation(self):
        old_len = litellm.token_counter(self._model, messages=self._conversation)

        self._conversation = trim_messages(self._conversation, self._model)

        new_len = litellm.token_counter(self._model, messages=self._conversation)
        if old_len != new_len:
            self._broadcast(
                "on_warn", f"Trimming conversation from {old_len} to {new_len} tokens."
            )

    def _add_function_results_to_conversation(self, tool_calls):
        # response_message["role"] = "assistant"
        # tool_calls = response_message["tool"]
        try:
            for tool_call in tool_calls:
                function_response = self._make_call(tool_call)
                function_response = sandwich_tokens(
                    function_response, self._model, self._max_call_response_tokens, 0.5
                )
                response = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_call["function"]["name"],
                    "content": function_response,
                }
                self._conversation.append(response)
        except Exception as e:
            # Warning: potential infinite loop if the LLM keeps sending
            # the same bad call.
            self._broadcast(
                "on_error", f"An exception occured while processing tool calls: {e}"
            )
