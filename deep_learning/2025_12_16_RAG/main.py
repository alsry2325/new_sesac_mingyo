import module as md
import asyncio
from prompts import initial_input,prompt_chain 


#랭체인기본
# md.Langchain_basic("LangChain이 무엇인가요?")
#llm예전 호출방식, 최신 호출방식 
# md.llm_basic_1("한 문장으로 인공지능을 설명해 보세요")

# md.llm_basic_2("프롬프트 엔지니어링")

# md.llm_basic_3("물김치")

# md.llm_basic_4("볶음밥")

# md.llm_basic_5("ChatGPT는 그저그래")

# md.llm_basic_6()
# md.llm_basic_7()

# md.run_router_workflow("1더하기2는 뭐지?")

# md.prompt_chain_workflow_2(initial_input, prompt_chain)
#비동기형식

asyncio.run(md.main())