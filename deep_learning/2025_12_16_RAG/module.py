from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from openai import OpenAI,AsyncOpenAI
from typing import List
#비동기
import asyncio
load_dotenv("C:/Users/USER/Desktop/apikeys.txt")
client = OpenAI()
#비동기방식
async_client = AsyncOpenAI()

def Langchain_basic(input):
    #1. 컴포넌트 정의 : prompt, model, parser 를 결정
    #2. 체인 생성 : 체인 만들기  chain = prompt | model | parser 
    #3. 체인 실행 : invoke, batch  chain.invoke({‘input’: ‘사용자 질문’})
    ##문장 길이수 max_tokens
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    response = llm.invoke(input,max_tokens=80)
    print(response.content)

def llm_basic_1(input):
    
    messages = [
    {
        "role": "system",
        "content": "당신은 불친절한 한국어 튜터입니다.",
    },
        {"role": "user", "content": input},
    ]
    #예전 방식(채팅 전용)
    response1 = client.chat.completions.create(model="gpt-4o", messages=messages)
    #최신 방식(채팅 / 단일 프롬프트 / 이미지 / 오디오 / 툴 호출 전부 포함)
    response2 = client.responses.create(model="gpt-4o", input=messages, max_output_tokens=60)

    print(f'{response1.choices[0].message.content}\n{response2.output_text}')

def llm_basic_2(input):

    messages=[
        {"role":"user","content":input},
    ]

    response = client.responses.create(model="gpt-4o-mini",input = messages) #최신은 messages가 아니라  input
    print(response.output_text)

#Chat Completions 방식 (구 API)
#Single-shot Prompting
#Prompt Template + 함수 래핑
def llm_basic_3(dish2 : str) -> str:
    prompt = ''' 다음 요리의 레시피를 생각해 주세요.
                요리명 : """{dish}"""
            '''
    messages = [{'role' : 'user', 'content' : prompt.format(dish = dish2)}]

    #옛날방식 호출 해보기
    response = client.chat.completions.create(model = 'gpt-4o-mini',messages = messages)
    recipe = response.choices[0].message.content 
    print(recipe)

def llm_basic_4(dish):
    
    system_prompt = """
                    사용자가 입력한 요리의 레시피를 생각해주세요.

                    출력은 다음 JSON 형식으로 해주세요 
                    ```
                    {
                    "재료" : ["재료1","재료2"],
                    "순서" : ["순서1","순서2"],
                    }
                    ```
                    """
    messages = [
        {"role" : "system" , "content" : system_prompt},
        {"role" : "user" , "content" : dish}
    ]
    response = client.responses.create(model="gpt-4o-mini",input = messages)
    print(response.output_text)

def llm_basic_5(input):

    messages = [
        {"role" : "system" , "content" : '입력을 긍정적, 부정적, 중립 중 하나로 분류하세요'},
        {"role" : "user" , "content" : input}
    ]
    #최신방식 호출
    response = client.responses.create(model="gpt-4o-mini",input = messages)
    print(response.output_text)

#Few-shot 기법
def llm_basic_6():
    messages=[
        {"role": "system", "content": "입력이 AI와 관련이 있는지 답변해 주세요."},
        {"role": "user", "content": "AI의 진화는 대단하다"},
        {"role": "assistant", "content": "true"},
        {"role": "user", "content": "오늘은 날씨가 좋다"},
        {"role": "assistant", "content": "false"},
        {"role": "user", "content": "날씨가 매우 좋다하다"},
    ]
    response = client.chat.completions.create(model='gpt-4o', messages= messages)
    print(response.choices[0].message.content)

#Zero-shot Chain-of-Thought
def llm_basic_7():
    messages=[
        {"role": "system", "content": "단계별로 생각해 주세요."},
        {"role": "user", "content": "10 + 2 * 3 - 4 * 2"},
    ]

    response = client.chat.completions.create(model='gpt-4o-mini', messages= messages)
    print(response.choices[0].message.content)

#2.Routing

def llm_call(prompt: str, model : str = "gpt-4o-mini") -> str : 
    messages = [] 
    messages.append({"role" : "user", "content" : prompt})
    chat_completion = client.chat.completions.create(
            model=model, 
            messages=messages
        )
    return chat_completion.choices[0].message.content
#Prompt Chaining  이전 응답 → 다음 프롬프트의 문맥(Context)
def prompt_chain_workflow_2(initial_input: str, prompt_chain: List[str]) -> List[str] :
    
    response_chain = [] 
    response = initial_input 
    for i, prompt in enumerate(prompt_chain) : 
        print(f'\n == 단계 {i} == \n')
        final_prompt = f'{prompt} \n\n 문맥(Context) : \n {response} \n 사용자 입력 : {initial_input} \n'
        print(f'프롬프트 : \n {final_prompt} \n')
        response = llm_call(final_prompt)
        response_chain.append(response)
    print(response_chain[-1])

#라우팅
def run_router_workflow(user_prompt:str):
        
    router_prompt = f'''
            사용자의 프롬프트/질문: {user_prompt}
            각 모델은 서로 다른 기능을 가지고 있습니다 사용자의 질문에 가장 적합한 모델으 선택하세요.
            -gpt-4o: 일반적인 작업에 가장 적합한 모델(기본 값)
            -o1-mini: 코딩 및 복잡한 문제 해결에 적합한 모델
            -gpt-4o-mini: 간단한 사칙연산 등의 작업에 적합한 모델

            모델명만 단답형으로 응답하세요
        '''
    selected_model =  llm_call(router_prompt)
    print('llmdl 선택한 모델',selected_model)
    response =llm_call(user_prompt,model = selected_model)
  
    return print(response)

#비동기 형식 챗봇답변

async def llm_call_async(prompt: str, model : str = "gpt-4o-mini") -> str : 
  messages = [] 
  messages.append({"role" : "user", "content" : prompt})
  chat_completion = await async_client.chat.completions.create(
      model=model, 
      messages=messages
      )
  return chat_completion.choices[0].message.content

async def run_llm_parallel(prompt_details):
    tasks = [llm_call_async(prompt['user_prompt'], prompt['model']) for prompt in prompt_details]
    responses = []
    for task in asyncio.as_completed(tasks):
        result = await task
        responses.append(result)
    return responses

async def main():
  question = ("아래 문장을 자연스러운 한국어로 번역해줘:\n"
    "\"Do what you can, with what you have, where you are.\" — Theodore Roosevelt")
  parallel_prompt_details = [
          {"user_prompt": question, "model": "gpt-4o"},
          {"user_prompt": question, "model": "gpt-4o-mini"},
          {"user_prompt": question, "model": "o3"},
      ]
  responses = await run_llm_parallel(parallel_prompt_details)
  aggregator_prompt = ("다음은 여러 개의 AI 모델이 사용자 질문에 대해 생성한 응답입니다.\n"
                          "당신의 역할은 이 응답들을 모두 종합하여 최종 답변을 제공하는 것입니다.\n"
                          "일부 응답이 부정확하거나 편향될 수 있으므로, 신뢰성과 정확성을 갖춘 응답을 생성하는 것이 중요합니다.\n\n"
                          "사용자 질문:\n"
                          f"{question}\n\n"
                          "모델 응답들:")
  for i  in range(len(parallel_prompt_details)):
    aggregator_prompt +=f'\n{i+1} 모델 이름: {parallel_prompt_details[i]["model"]} , 모델 응답:{responses[i]}\n'

  print(aggregator_prompt)
  final_resp = await llm_call_async(aggregator_prompt,model='gpt-4o')
  print(final_resp)