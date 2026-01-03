import module as md
import prompts  as pm
import json
import asyncio
# 1. OpenAI 평가 알고리즘 사용법:
# LLM의 응답을 기준 정답이나 규칙과 비교해 자동으로 품질을 평가한다.

# 2. LangChain 기본 사용법:
# 프롬프트(prompt), LLM, 출력 파서(parser)를 조합해 LLM 호출 파이프라인을 구성한다.

# 3. Runnable 객체:
# prompt | llm | parser 형태로 연결해 실행 가능한 단일 객체로 만든다.

# 4. Batch 사용법:
# 여러 입력을 한 번에 묶어 처리해 호출 비용과 지연 시간을 줄인다.

# 5. Chain 만드는 법:
# 여러 Runnable을 순서대로 연결해 입력부터 출력까지의 처리 흐름을 정의한다.

# 6. Stream 사용법:
# LLM 응답을 토큰 단위로 실시간 받아 즉시 출력하거나 후처리한다.

# 7. Parser 사용법:
# LLM의 자유로운 텍스트 출력을 JSON이나 객체 같은 구조화된 결과로 변환한다.

#응답 객체 생성
orchestrator_response = md.llm_call(pm.orchestrator_prompt, model = 'gpt-4o')
response_json=json.loads(orchestrator_response.replace('```json', '').replace('```',''))
print(response_json) 
#analysis만 추출
analy_sis = response_json.get('analysis','')
print(analy_sis)
#subtasks만 추출
sub_tasks = response_json.get('subtasks',[])
print(sub_tasks[0])
# 여러 하위 작업(sub_tasks)에 대해 워커 LLM이 처리할 개별 프롬프트를 생성한다
worker_prompts = [pm.get_worker_prompt(pm.user_query,task['sub_question'],task['description'])for task in sub_tasks]
print(worker_prompts)

worker_responses = await md.run_llm_parallel(worker_prompts)
# 3개의 답변이 있음
print(len(worker_responses))
print(worker_responses[0])


#Evaluation-optimizer

summary = md.llm_call(pm.user_query,model='gpt-3.5-turbo')
print(summary)

#평가 프롬포트
print(pm.evaluator_prompt)

#평가, 요약
final_evaluation_prompt = md.llm_call(pm.evaluator_prompt+summary) 
print(final_evaluation_prompt)

#이미 생성된 평가 결과를 다시 입력으로 사용
evaluation_result =  md.llm_call(final_evaluation_prompt, model='gpt-4o')

print(evaluation_result)
print(summary)
print(pm.user_query)

retries = 1
pm.user_query += f"{retries}차 요약 결과: \n\n{summary}\n"
pm.user_query += f"{retries}차 요약 피드백: \n\n{evaluation_result}\n"
print(pm.user_query)

#예제문제 
#1.summary 얻어내기
#2.final_evaluator_prompt =  
#3.evaluation_result
#4.평가 결과 = PASS/ FAIL -> 프로그램 종료 / 계속 결정
#5. user_query+= 요약 결과, 피드백
md.loop_workflow(pm.user_query,pm.evaluator_prompt)

 
#2. LangChain 기본 사용법
#라이브러리 가상환경에 셋팅
#pip install langchain-openai langgraph langchain-text-splitters python-dotenv pydantic langchain_community faiss-cpu retry langchain-chroma

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel,Field
 

model = ChatOpenAI(model='gpt-4o-mini',temperature=0)

messages = [SystemMessage("You are a helpful assistant"),
            HumanMessage("안녕하세요! 저는 곤이라고 합니다"),
            AIMessage("안녕하세요, 곤님 어떤 도움이 필요하신가요"),
            HumanMessage('제이름을 아시나요?')]
# Runnable을 실행하여 입력에 대한 최종 결과를 반환한다
# Runnable : 입력을 받아 실행(invoke)할 수 있는 실행 단위 객체
ai_message = model.invoke(messages)
print(ai_message.content)

for chunk in model.stream(messages):
  # flush=True 출력 버퍼를 즉시 비우고 바로 화면에 출력하라
  print(chunk.content, end="",flush=True)

#실행 시 값으로 채워주는 템플릿 객체 재사용하기 위해
prompt = PromptTemplate.from_template('다음 요리의 레시피를 생각해 주세요. 요리명:{dish}')

prompt_value = prompt.invoke({'dish':'라면'})
print(prompt_value.text)

ai_message = model.invoke(prompt_value)
print(ai_message.content)

#역할 구분
prompt = ChatPromptTemplate.from_messages([
    ('system','사용자가 입력한 요리의 레시피를 생각해주세요.'),
    ('human','{dish}')
])

prompt_value = prompt.invoke({'dish':'라면'})
ai_message = model.invoke(prompt_value)
print(ai_message.content)

#MessagesPlaceholder  기억 히스토리

prompt = ChatPromptTemplate.from_messages([('system', 'You are a helpful assistant.'),
                                           MessagesPlaceholder('chat_history', optional=True),
                                           ('human', '{input}')])

prompt_value = prompt.invoke({'chat_history':[
            HumanMessage("안녕하세요! 저는 곤이라고 합니다"),
            AIMessage("안녕하세요, 곤님 어떤 도움이 필요하신가요")],'input':'제이름을 아시나요?'})
print(prompt_value)

ai_message = model.invoke(prompt_value)
print(ai_message)

#출력을 문자열로 고정  Parser()
output_parser = StrOutputParser()
ai_message = output_parser.invoke(ai_message)
#print(ai_message.content)
print(ai_message)

#파이프라인 구현 LCEL
#Prompt, LLM, Parser 등 Runnable들을 | 연산자로 조합해 실행 흐름을 선언적으로 표현하는 문법이다.
model = ChatOpenAI(model='gpt-4o-mini',temperature=0)
prompt = ChatPromptTemplate.from_messages([('system', 'You are a helpful assistant.'),
                                           MessagesPlaceholder('chat_history', optional=True),
                                           ('human', '{input}')])

#파이프라인 프롬프트 -> 모델 -> 결과
chain = prompt| model|output_parser

chain.invoke({'input':'태양의 공전 속도는?'})
#MessagesPlaceholder 활용
chain.invoke({'chat_history':[
            HumanMessage("안녕하세요! 저는 곤이라고 합니다"),
            AIMessage("안녕하세요, 곤님 어떤 도움이 필요하신가요")],'input':'제이름을 아시나요?'})


#{한국어_단어}를 영어로 번역합니다
prompt1 = ChatPromptTemplate.from_template('translates{korean_word}to English')
#옥스퍼드 사전을 사용하여 {영어_단어}를 한국어로 설명해 주세요.'
prompt2 = ChatPromptTemplate.from_template('= {english_word}나오면  한국어로 번역해줘 의미도 함께')

llm = ChatOpenAI(model='gpt-3.5-turbo-0125')

chain1 = prompt1 | llm |StrOutputParser()

chain1.invoke({'korean_word':'내일'})

#prompt1요청이 나온다음  prompt2 프롬프트요청대로  실행해라
chain2 = {'english_word': chain1} | prompt2 | llm |StrOutputParser()

chain2.invoke({'korean_word': '내일'})


prompt = ChatPromptTemplate.from_template(
    '지구과학에서 {topic} 에대해 간단히 설명하시오'
)

ouput_parser = StrOutputParser()
llm = ChatOpenAI(model ='gpt-4o')
chain = prompt | llm | output_parser

chain.invoke({'topic':'지구 자전'})

#동일한 Runnable을 여러 입력에 대해 반복 실행해 결과 리스트를 반환하는 실행 방식이다.
#batch
topics =['지구 공전 ','대륙이동','화산 활동']
results = chain.batch([{'topic':t}for t in topics])

for i  in results:
  print(i)

#실시간 
stream = chain.stream({'topic':'지진'})

for i  in stream:
  print(i,end='')


examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ('human', '{input}'),
        ('ai', '{output}')
    ]
)
#예시 기반(few-shot) 대화를 시스템적으로 넣는 방법
#여러 개의 대화 예시(입력–출력 쌍)를 채팅 메시지 형태로 프롬프트에 삽입하는 템플릿
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ('system','당신은 과학과 수학에 대해 잘 아는 교육자다'),
        few_shot_prompt,
        ('human','{input}')
    ]
)

ouput_parser = StrOutputParser()
llm = ChatOpenAI(model ='gpt-4o')
chain = final_prompt | llm | output_parser

chain.invoke('지구의 자전속도는? 짧게 말해')

#csv 파서
output_parser = CommaSeparatedListOutputParser()
#지침까지 가져왔음
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

prompt = PromptTemplate(
    template = 'List five {subject}.\n{format_instructions}',
    input_variables=['subject'],
    partial_variables={'format_instructions':format_instructions},
)

llm = ChatOpenAI(model='gpt-4o',temperature= 0)

chain = prompt| llm | output_parser

chain.invoke({'subject':'popular Korean cuisine'})


#json output파서
#LLM의 출력을 JSON으로 강제하고, 그 JSON을 Pydantic 모델로 파싱까지 하는 정석 패턴
class CuisineRecipe(BaseModel):
    name: str = Field(description ="name of a cuisine")
    recipe: str = Field(description='recipe to cook the cuisine')

output_parser = JsonOutputParser(pydantic_object=CuisineRecipe)

format_instructions = output_parser.get_format_instructions()

print(format_instructions)

prompt = PromptTemplate(template='Answer the user query. \n{format_instructions}\n{query}\n',
                        input_variables =["query"],
                        partial_variables={'format_instructions':format_instructions})
## Pydantic 모델 기반 JSON 출력을 강제하고 객체로 파싱하는 체인 구성
chain = prompt| llm |output_parser

chain.invoke({'query':'비빔밥 레시피 알려줘'})