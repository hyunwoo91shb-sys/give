import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

# =============================================
# ✅ PDF 파일명만 여기서 수정하세요
PDF_FILE_PATH = "증여법관련.pdf"
# =============================================

# Step 1: PDF 문서 기반 답변 시도
RAG_PROMPT = """
당신은 증여 관련 전문 Q&A 챗봇입니다.
아래의 참고 문서(context)를 바탕으로 사용자의 질문에 답변해주세요.

다음 경우에는 반드시 "RAG_INSUFFICIENT" 한 단어만 출력하세요:
- 참고 문서에 질문과 관련된 내용이 없거나 너무 부족한 경우
- 참고 문서만으로는 정확한 답변이 어려운 경우

그 외에는 문서를 바탕으로 정확하고 친절하게 답변해주세요.

[참고 문서]
{context}

[질문]
{question}
"""

# Step 2: GPT 자체 지식으로 답변 시도
GPT_FALLBACK_PROMPT = """
당신은 증여 관련 전문 Q&A 챗봇입니다.
사용자의 질문에 대해 증여세법 및 관련 세법 지식을 바탕으로 답변해주세요.

다음 경우에는 반드시 "GPT_INSUFFICIENT" 한 단어만 출력하세요:
- 질문이 증여와 전혀 관련 없는 경우
- 확실한 정보가 없어 잘못된 답변을 드릴 가능성이 높은 경우
- 질문이 너무 개인적이거나 구체적인 사례라 일반적 답변이 불가능한 경우

그 외에는 친절하게 답변하고, 마지막에 아래 문구를 추가해주세요:
"📌 위 내용은 일반적인 세법 기준이며, 개인 상황에 따라 다를 수 있습니다. 정확한 사항은 세무사 또는 금융 전문가에게 확인하시길 권장합니다."

[질문]
{question}
"""

UNABLE_TO_ANSWER = (
    "죄송합니다. 해당 질문에 대해서는 정확한 답변을 드리기 어렵습니다. 😔\n\n"
    "더 정확한 도움을 받으시려면 **세무사** 또는 **금융 전문가**에게 문의해 주세요."
)

# 채팅창 첫 인사 (배너와 중복되지 않게 간단하게)
WELCOME_MESSAGE = "안녕하세요! 👋 무엇이든 편하게 질문해 주세요 😊"

QUICK_QUESTIONS = [
    "미성년자 자녀, 증여 공제 한도는 얼마일까?",
    "성인 자녀, 증여 공제 한도는 얼마일까?",
    "증여 꿀팁 알려주세요",
]


@st.cache_resource
def load_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def set_background(image_path: str, opacity: float = 0.15):
    import base64
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:{mime};base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                inset: 0;
                background: rgba(255, 255, 255, {1 - opacity});
                z-index: 0;
            }}
            .stApp > * {{
                position: relative;
                z-index: 1;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        pass


def init_page():
    st.set_page_config(page_title="증여 Q&A 챗봇", page_icon="🤗")
    st.header("증여 Q&A 챗봇 🤗")
    st.sidebar.title("Options")

    # =============================================
    # ✅ 배경 이미지 파일명과 투명도를 여기서 수정하세요
    set_background("background.png", opacity=0.2)
    # =============================================

    # 항상 상단에 고정되는 안내 배너
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(232,244,253,0.92), rgba(240,247,255,0.92));
            border-left: 4px solid #4a90d9;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 16px;
            font-size: 0.95rem;
            line-height: 1.7;
        ">
            아이에게 용돈도 주고, 예금·펀드 같은 금융 상품도 가입해 주셨군요?
            정말 든든한 부모님이세요 😊
            <br><br>
            혹시 <strong>증여세 신고</strong>나 <strong>공제 한도</strong>,
            <strong>절세 방법</strong> 같은 게 궁금하신가요?<br>
            궁금한 것은 무엇이든 편하게 물어봐 주세요!
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_model(temperature=0):
    models = ("gpt-5.5", "gpt-5.4-mini")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "gpt-5.5":
        return ChatOpenAI(temperature=temperature, model="gpt-5.5")
    else:
        return ChatOpenAI(temperature=temperature, model="gpt-5.4-mini")


def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_messages([("user", RAG_PROMPT)])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_gpt_fallback_chain(llm):
    prompt = ChatPromptTemplate.from_messages([("user", GPT_FALLBACK_PROMPT)])
    return prompt | llm | StrOutputParser()


def init_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
    if "quick_question" not in st.session_state:
        st.session_state.quick_question = None
    if "show_quick_buttons" not in st.session_state:
        st.session_state.show_quick_buttons = True


def handle_chat(user_input, rag_chain, gpt_fallback_chain):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        # Step 1: PDF RAG 시도
        with st.spinner("📄 문서에서 답변을 찾는 중..."):
            rag_response = rag_chain.invoke(user_input)

        if "RAG_INSUFFICIENT" not in rag_response:
            response = rag_response
            st.markdown(response)

        else:
            # Step 2: GPT 자체 지식으로 fallback
            with st.spinner("🤖 추가 지식으로 답변을 생성하는 중..."):
                gpt_response = st.write_stream(
                    gpt_fallback_chain.stream({"question": user_input})
                )

            if "GPT_INSUFFICIENT" in gpt_response:
                response = UNABLE_TO_ANSWER
                st.markdown(response)
            else:
                response = gpt_response

    st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    init_page()
    init_messages()

    llm = select_model()

    with st.spinner("📄 PDF 문서를 불러오는 중..."):
        try:
            vectorstore = load_vectorstore(PDF_FILE_PATH)
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            rag_chain = build_rag_chain(llm, retriever)
            gpt_fallback_chain = build_gpt_fallback_chain(llm)
            st.sidebar.success(f"✅ 문서 로드 완료\n`{PDF_FILE_PATH}`")
        except FileNotFoundError:
            st.error(f"❌ PDF 파일을 찾을 수 없습니다: `{PDF_FILE_PATH}`\n\n상단 `PDF_FILE_PATH` 변수를 수정해주세요.")
            st.stop()
        except Exception as e:
            st.error(f"❌ 오류 발생:\n```\n{traceback.format_exc()}\n```")
            st.stop()

    # 채팅 히스토리 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 빠른 질문 버튼
    if st.session_state.show_quick_buttons:
        st.markdown("##### 💬 이런 것들이 궁금하신가요?")
        cols = st.columns(len(QUICK_QUESTIONS))
        for i, (col, question) in enumerate(zip(cols, QUICK_QUESTIONS)):
            with col:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    st.session_state.quick_question = question
                    st.rerun()

    # 빠른 질문 버튼 클릭 처리
    if st.session_state.quick_question:
        question = st.session_state.quick_question
        st.session_state.quick_question = None
        handle_chat(question, rag_chain, gpt_fallback_chain)

    # 직접 입력 처리
    if user_input := st.chat_input("증여 관련 질문을 입력하세요..."):
        st.session_state.show_quick_buttons = False
        handle_chat(user_input, rag_chain, gpt_fallback_chain)


main()
