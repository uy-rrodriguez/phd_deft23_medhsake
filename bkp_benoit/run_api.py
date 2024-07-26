from api_keys import DEEPL_TOKEN, OPENAI_TOKEN, COHERE_TOKEN, AI21_TOKEN

import openai
import backoff
openai.api_key = OPENAI_TOKEN

openai_lm_models = ['text-davinci-003', 'text-davinci-002', 'code-davinci-002', 'code-cushman-001', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'davinci', 'curie', 'babage', 'ada']
openai_chat_models = ['gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314']

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def openai_complete(prompt, model='text-davinci-003'):
    result = openai.Completion.create(model=model, prompt=prompt, temperature=0, max_tokens=32)
    return result['choices'][0]['text']

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def openai_chat(prompt, model='gpt-3.5-turbo'):
    result = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=32)
    return result['choices'][0]['message']['content']

import cohere, requests, time
cohere_client = cohere.Client(COHERE_TOKEN)
cohere_models = ['command-xlarge-beta', 'command-xlarge-nightly', 'xlarge', 'medium', 'command-medium-beta', 'command-medium-nightly']
@backoff.on_exception(backoff.expo, requests.exceptions.RetryError)
def cohere_complete(prompt, model='command-xlarge-beta'):
    response = cohere_client.generate(model=model, prompt=prompt, max_tokens=32, temperature=1, k=0, p=0.75, stop_sequences=[], return_likelihoods='NONE')
    time.sleep(20) # max 3 queries per minute (free account)
    return response.generations[0].text

import ai21
ai21.api_key = AI21_TOKEN

ai21_models = ['j1-jumbo', 'j1-grande', 'j1-grande-instruct', 'j1-large']
def ai21_complete(prompt, model='j1-jumbo'):
    result = ai21.Completion.execute(model=model, prompt=prompt, maxTokens=32, temperature=0.5, numResults=1, topP=0.1)
    return result['completions'][0]['data']['text']

def main(result_path: str, corpus_path: str, model: str = 'openai/gpt-3.5-turbo-0301', template_num: int = 0):
    api, llm = model.split('/', 1)

    def generate(input_string):
        if api == 'openai':
            if llm in openai_chat_models:
                #print('This is a chat model')
                return openai_chat(input_string, llm)
            else:
                return openai_complete(input_string, llm)
        elif api == 'cohere':
            return cohere_complete(input_string, llm)
        elif api == 'ai21':
            return ai21_complete(input_string, llm)

    import deft
    results = deft.run_inference(generate, corpus_path, deft.lm_templates[template_num])
    deft.write_results(results, result_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
