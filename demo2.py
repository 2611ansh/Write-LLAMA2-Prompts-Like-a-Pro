from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

#######------------- LLM-------------#######
my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com",
    "token" : "skills-network"
}

params = {
        GenParams.MAX_NEW_TOKENS: 256, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-2-70b-chat', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",
        )

llm = WatsonxLLM(model=LLAMA2_model)

template = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
<</SYS>>
Current conversation: {history}
[/INST]  </s><s>[INST] user input: {input} 
AI Assistant:
[/INST]
"""
    
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)

response1 = conversation.predict(input="How is like to live in Toronto")
print("first response:", response1)

response2 = conversation.predict(input="tell me more about it")
print("second response:", response2 )