from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


class LLMClient:

    def __init__(
        self,
        api_key,
        model="openai/gpt-oss-120b",
        temperature=0.1
    ):

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=api_key
        )

    def generate(self, context, question):

        system_prompt = """
You are a helpful AI document assistant.

Answer the user's question using only the information
provided in the context.

If the answer is not available in the context,
clearly say that the information is not available
in the provided documents.

Do not make up information.
"""

        user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)

        return response.content