from langchain.base_language import BaseLanguageModel
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from loader import DialogueLoader
from chains.dialogue_answering.prompts import (
    DIALOGUE_PREFIX,
    DIALOGUE_SUFFIX,
    SUMMARY_PROMPT
)

# 定义了一个名为DialogueWithSharedMemoryChains的类
# 包含了对话系统的相关组件和功能。主要功能如下：
# 类变量和实例变量用于存储相关信息和配置。
# __init__方法用于初始化类的实例，接收零-shot回应语言模型和问答语言模型作为参数。
# _init_cfg方法用于配置语言模型。
# _init_state_of_history方法用于初始化对话历史状态，加载对话并创建文本分割器和向量存储。
# _agents_answer方法用于创建会话缓存内存和只读共享内存，以及创建LLMChain实例。
# _create_agent_chain方法用于创建代理链，包括工具和提示模板的定义。
# 此代码用于构建一个具有共享内存的对话系统，其中包括零-shot回应和问答功能。
class DialogueWithSharedMemoryChains:
    zero_shot_react_llm: BaseLanguageModel = None
    ask_llm: BaseLanguageModel = None
    embeddings: HuggingFaceEmbeddings = None
    embedding_model: str = None
    vector_search_top_k: int = 6
    dialogue_path: str = None
    dialogue_loader: DialogueLoader = None
    device: str = None

    def __init__(self, zero_shot_react_llm: BaseLanguageModel = None, ask_llm: BaseLanguageModel = None,
                 params: dict = None):
        # 初始化实例变量
        self.zero_shot_react_llm = zero_shot_react_llm
        self.ask_llm = ask_llm
        params = params or {}
        self.embedding_model = params.get('embedding_model', 'GanymedeNil/text2vec-large-chinese')
        self.vector_search_top_k = params.get('vector_search_top_k', 6)
        self.dialogue_path = params.get('dialogue_path', '')
        self.device = 'cuda' if params.get('use_cuda', False) else 'cpu'

        # 创建对话加载器实例
        self.dialogue_loader = DialogueLoader(self.dialogue_path)
        # 初始化配置
        self._init_cfg()
        # 初始化历史状态
        self._init_state_of_history()
        # 创建记忆链和共享内存
        self.memory_chain, self.memory = self._agents_answer()
        # 创建代理链
        self.agent_chain = self._create_agent_chain()

    def _init_cfg(self):
        # 配置语言模型
        model_kwargs = {
            'device': self.device
        }
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs=model_kwargs)

    def _init_state_of_history(self):
        # 加载对话
        documents = self.dialogue_loader.load()
        # 创建文本分割器
        text_splitter = CharacterTextSplitter(chunk_size=3, chunk_overlap=1)
        # 将对话切分为文本块
        texts = text_splitter.split_documents(documents)
        # 创建Chroma向量存储
        docsearch = Chroma.from_documents(texts, self.embeddings, collection_name="state-of-history")
        # 创建RetrievalQA实例
        self.state_of_history = RetrievalQA.from_chain_type(llm=self.ask_llm, chain_type="stuff",
                                                            retriever=docsearch.as_retriever())

    def _agents_answer(self):
        # 创建会话缓存内存和只读共享内存
        memory = ConversationBufferMemory(memory_key="chat_history")
        readonly_memory = ReadOnlySharedMemory(memory=memory)
        # 创建LLMChain实例
        memory_chain = LLMChain(
            llm=self.ask_llm,
            prompt=SUMMARY_PROMPT,
            verbose=True,
            memory=readonly_memory,  # 使用只读内存以防止工具修改内存
        )
        return memory_chain, memory

    def _create_agent_chain(self):
        # 获取对话参与者
        dialogue_participants = self.dialogue_loader.dialogue.participants_to_export()
        # 创建工具列表
        tools = [
            Tool(
                name="State of Dialogue History System",
                func=self.state_of_history.run,
                description=f"Dialogue with {dialogue_participants} - The answers in this section are very useful "
                            f"when searching for chat content between {dialogue_participants}. Input should be a "
                            f"complete question. "
            ),
            Tool(
                name="Summary",
                func=self.memory_chain.run,
                description="useful for when you summarize a conversation. The input to this tool should be a string, "
                            "representing who will read this summary. "
            )
        ]

        # 创建ZeroShotAgent的提示模板
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=DIALOGUE_PREFIX,
            suffix=DIALOGUE_SUFFIX,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )

        # 创建LLMChain实例和代理
        llm_chain = LLMChain(llm=self.zero_shot_react_llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=self.memory)

        return agent_chain
