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

# Create two prompt templates: one for generating a random question about a given topic, and another for answering a given question
pt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a random question about {topic}: Question: ")
# Create an LLM chain with the LLAMA2 model and the first prompt template
prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

# Run the chain with the input "cat", which will generate a random question about "cat" and then answer that question
result = prompt_to_LLAMA2.run("cat")
print(result)

temp = """
<s>[INST] <<SYS>>
Generate a random question about: 
<</SYS>>

Question: {user_message} [/INST]
"""
pt = PromptTemplate(
    input_variables=["user_message"],
    template= temp)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)
result = prompt_to_LLAMA2.run("cat")
print(result)