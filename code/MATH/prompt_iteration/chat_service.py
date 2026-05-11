import openai
import os
import time
from prompt_lib import MMLU_QUESTION, TEMPERATURE, MAX_TOKENS

# from prompt_iteration.keys import Keys


OPENAI_MAX_RETRIES = max(1, int(os.getenv("OMAC_OPENAI_MAX_RETRIES", "3")))
OPENAI_RETRY_SLEEP = float(os.getenv("OMAC_OPENAI_RETRY_SLEEP", "2"))


def retryable_openai_errors():
    names = ("APIError", "APIConnectionError", "RateLimitError", "Timeout", "ServiceUnavailableError")
    return tuple(
        error_type
        for name in names
        for error_type in [getattr(openai.error, name, None)]
        if error_type is not None
    )


class ChatService:
    # _keys = Keys

    def __init__(self, system=None, keys=None):
        self.dialog = []
        # self.keys = keys or self._keys
        if system:
            self.dialog.append({"role": "system", "content": system})

    def ask(self, question, model):
        self.dialog.append({"role": "user", "content": question})
        for attempt in range(OPENAI_MAX_RETRIES):
            try:
                resp = openai.ChatCompletion.create(
                          model=model,
                          temperature=TEMPERATURE,
                          messages=self.dialog,
                          max_tokens=MAX_TOKENS,
                          n=1)  # TODO: check the meaning of n
                break
            except retryable_openai_errors():
                if attempt == OPENAI_MAX_RETRIES - 1:
                    raise
                time.sleep(OPENAI_RETRY_SLEEP * (attempt + 1))
        self.dialog.append(resp['choices'][0]['message'])
        return resp['choices'][0]['message']['content']


if __name__ == '__main__':
    service = ChatService('You are a Chinese poet.')
    print(service.ask('"举头望明月"的下一句是？'))
    print(service.ask('这首诗是？'))
