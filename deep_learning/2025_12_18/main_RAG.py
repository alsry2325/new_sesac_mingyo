from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#오늘 학습한 내용
'''
Hugging Face & Facebook AI 개요

PyPDFLoader 사용법 (PDF 파일 로딩)

CSVLoader 사용법 (CSV 데이터 로딩)

DirectoryLoader 사용법 (폴더 단위 문서 로딩)

TextLoader 사용법 (TXT 파일 읽기)

문서 로더별 활용 차이 정리

메모리 사용 개념

채팅 메모리(Chat History / 대화 기억)
'''

load_dotenv("C:/Users/USER/Desktop/apikeys.txt")

url = 'https://ko.wikipedia.org/wiki/위키백과:정책과_지침'

#LangChain에서 웹페이지(URL)의 내용을 가져와서 문서(Document) 형태로 로드하는 객체
loader =WebBaseLoader(url)
docsTest =loader.load()
print(docsTest)

# 1. 문서 분할 (Character 단위)
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
docs = text_splitter.split_documents(docsTest)
print(docs)

# 2. 임베딩 모델  문서 검색 / RAG 
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 3. VectorStore 생성 (Chroma) 
db = Chroma.from_documents(docs, embeddings)

# 4. Retriever 생성
retriever = db.as_retriever()

# 5. 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template(
    """
    다음 문맥을 바탕으로 질문에 답변해 주세요.

    문맥:
    {context}

    질문:
    {question}
    """
)

# 6. LLM 모델
model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# 7. Document → 문자열 변환 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. RAG 체인 구성
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# 9. 실행
query = "외부링크에 말해봐"
result = rag_chain.invoke(query)
print(result)