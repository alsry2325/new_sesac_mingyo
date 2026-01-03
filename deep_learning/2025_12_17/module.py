from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import json
import asyncio

load_dotenv("C:/Users/USER/Desktop/apikeys.txt")

client = OpenAI()
async_client = AsyncOpenAI()
#LLM 호출 함수 (단일 프롬프트 → 단일 응답)
def llm_call(prompt: str, model : str = "gpt-4o-mini") -> str :
  messages = []
  messages.append({"role" : "user", "content" : prompt})
  chat_completion = client.chat.completions.create(
      model=model,
      messages=messages
      )
  return chat_completion.choices[0].message.content
# 단일 프롬프트를 OpenAI API로 비동기 호출하는 LLM 래퍼 함수
#래퍼 함수(Wrapper Function)란?
#복잡한 기능이나 외부 API를 감싸서 더 단순하고 사용하기 쉬운 형태로 제공하는 함수
async def llm_call_async(prompt: str, model : str = "gpt-4o-mini") -> str :
  messages = []
  messages.append({"role" : "user", "content" : prompt})
  chat_completion = await async_client.chat.completions.create(
      model=model,
      messages=messages
      )
  return chat_completion.choices[0].message.content
## 여러 프롬프트를 LLM에 비동기로 병렬 호출하고, 완료 순서대로 응답을 수집하는 함수
async def run_llm_parallel(prompt_list):
    tasks = [llm_call_async(prompt) for prompt in prompt_list]
    responses = []
    for task in asyncio.as_completed(tasks):
        result = await task
        responses.append(result)
    return responses

def loop_workflow(user_query,evaluator_prompt,max_retries =5):
   retries = 0
   #5번만 실행
   while retries < max_retries:
      #summary 얻어내기
      summary = llm_call(user_query, model='gpt-3.5-turbo')
      #평가기준, 요약기준 결과를 대입
      final_evaluation_prompt = evaluator_prompt+summary
      #이미 생성된 평가 결과를 다시 입력으로 사용
      evaluation_result = llm_call(final_evaluation_prompt, model='gpt-4o').strip()
      print(evaluation_result)

      if '평가결과 = PASS' in evaluation_result:
         print("통과 ! 최종요약이 승인됨")
         return summary
      
      retries+= 1
      print(f"재시도 필요({retries}/{max_retries})")
      if retries >= max_retries:
        print("최대 재시도 횟수 도달. 마지막 요약을 반환합니다")

      user_query += f"{retries}차 요약 결과: \n\n{summary}\n"
      user_query += f"{retries}차 요약 피드백: \n\n{evaluation_result}\n"
