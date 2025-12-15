from openai import OpenAI
from dotenv import load_dotenv

def load_api():
    
    load_dotenv("C:/Users/USER/Desktop/apikeys.txt")

def response(input):
    messages = [{'role':'system', 'content':'You are a helpful assistant.'},
            {'role':'user', 'content':input}]
    client = OpenAI() 
    response = client.chat.completions.create(model='gpt-4o', messages=messages)
    print(response)
    print("===============")
    response = list(response)
    #튜플리스트는 딕셔너리로 변환하고 접근해야한다.
    response_dict = dict(response)
    print(response_dict)
    print("==================")
    content = response_dict["choices"][0].message.content
    print(content)
    
    

