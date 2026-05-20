# encoding: utf-8
import requests

AIDC_URL = "http://your-real-url"  # 离线

AIDC_TOKEN = 'your-key'

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + AIDC_TOKEN
}

AIDC_MODEL_MAPPING = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1-2025-04-14-GlobalStandard",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14-GlobalStandard",
    "gpt-5": "gpt-5-2025-08-07-GlobalStandard",
    "gpt-image-1": "gpt-image-1",
    "o3-mini": "o3-mini-2025-01-31",
    "o4-mini": "o4-mini-2025-04-16-GlobalStandard",
    "o1-mini": "o1-mini-2024-09-12",
    "o1": "o1-2024-12-17",
    # claude
    "claude-35-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-37-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-4-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-Opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    # qwen
    "qwen-max": "qwen-max-2025-01-25",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen3-max": "qwen3-max",
    "qwen-vl-max": "qwen-vl-max",
    
    # deepseek
    "deepseek-v3": "deepseek-v3",
    "deepseek-r1": "deepseek-r1"

}



def send_request_by_prompt(prompt_text, model):
    model = AIDC_MODEL_MAPPING[model]
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(AIDC_URL, headers=headers, json=data)

    return response.json()["choices"][0]['message']['content']


def send_request_by_message(messages, model, temperature=None):
    model = AIDC_MODEL_MAPPING[model]
    data = {
        "model": model,
        "messages": messages
    }
    if temperature is not None:
        data["temperature"] = temperature

    response = requests.post(AIDC_URL, headers=headers, json=data)
    # print(response.json())
    return response.json()["choices"][0]['message']['content']

    
def completion(model, messages, temperature):
    try:
        model_name = AIDC_MODEL_MAPPING[model]
        data = {
            "model": model_name,
            "messages": messages
        }
        if temperature is not None:
            data["temperature"] = temperature
        
        response = requests.post(AIDC_URL, headers=headers, json=data)
        response_json = response.json()
        
        #  提取 usage 信息
        usage = response_json.get("usage", {})
        content = response_json["choices"][0]['message']['content']
        
        # 返回内容和 token 使用情况
        return content, usage
        
    except Exception as e:
        print(e)
        return None, {}
