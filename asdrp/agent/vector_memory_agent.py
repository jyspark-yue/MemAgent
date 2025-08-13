import asyncio
from typing import List
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent.workflow import FunctionAgent, AgentOutput

from llama_index.core.memory import Memory, VectorMemory

import chromadb
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.llms import ChatMessage


class vectorMemoryAgent:
    
    def __init__(self, k: int = 2, llm: LLM = OpenAI(model="gpt-4.1-mini"), tools: List[FunctionTool] = []):
        '''
        Initializes a vectorMemoryAgent, with k=2 and default LLM as OpenAI gpt 4.1 mini
        '''
        self.k = k
        self.llm = llm
        #initialize vector_memory
        client = chromadb.EphemeralClient()
        vstore = ChromaVectorStore(
            chroma_collection=client.get_or_create_collection("test_collection")
        )
        self.vector_memory = VectorMemory.from_defaults(
            vector_store=None, 
            embed_model=OpenAIEmbedding(),
            retriever_kwargs={"similarity_top_k":self.k}
        )
        self.agent = FunctionAgent(
            llm = self.llm,
            tools = tools
        )
    
    async def processMessage(self, message):
        '''
        Return's an agent's response to a user input
        '''
        response = await self.agent.run(message)
        return response

    async def aget(self, msg:str):
        '''
        Given a chat message, retrieve & return the top k most relevant past messages
        '''
        msgs = []
        for i in self.vector_memory.get(msg):
            msgs.append(i)
        return msgs
    
    async def aput(self, msg:str):
        '''
        Given a user input, put the user's chat + the agent's response into the vector memory database
        '''
        self.vector_memory.put(ChatMessage.from_str(msg, "user"))
        response = await self.processMessage(msg)
        print(f"Agent response:\n{response}")
        if isinstance(response, AgentOutput):
            self.vector_memory.put(ChatMessage.from_str(response.response.content, "assistant"))
        elif isinstance(response, ChatMessage):
            self.vector_memory.put(ChatMessage.from_str(response.content, "assistant"))
        else:
            self.vector_memory.put(ChatMessage.from_str(str(response), "assistant"))

    # the following functions are for testing purposes only

    async def userSetupVectorDatabase(self):
        '''
        Asks user for fact(s) to be put in the vector memory module
        Only puts user messages into vector memory as facts (does not store agent response)
        '''
        userInput = input("Enter a message: ")
        while userInput.strip():
            await self.aput(userInput)
            userInput = input("Enter a message: ")

    async def userTests(self):
        '''
        Designed for testing
        Asks user for prompt(s) / questions, agent will look for most relevant past messages in vector_memory
        Stores agent responses into vector memory after being tested
        '''
        test = input("Test: ")
        while test.strip():
            msgs = await self.aget(test)
            print(f"First {self.k} relevant messages: \n{msgs}")

            relevant = test + "\nPrevious messages: \n"
            for i in msgs:
                relevant += i.role + ": " + i.content + "\n"
            print(relevant)

            response = await self.processMessage(relevant)
            print(f"Response:\n{response}")

            test = input("\nTest: ")

    async def runBasicTest(self):
        '''
        Tests successful message storing & retrieval. 
        '''
        msgs = [
            ChatMessage.from_str("Bob likes dogs.", "user"),
            ChatMessage.from_str("Bob dislikes apples.", "user"),
            ChatMessage.from_str("Alice likes apples.", "user"),
        ]
        for msg in msgs:
            self.vector_memory.put(msg)

        # look for successful message retrieval
        testMsg = await self.aget("What does Bob like?")
        if len(testMsg) == self.k:
            print("Basic test successful")

        #clears vector_memory
        #self.vector_memory.reset()

async def main():
    # run basic test
    ag = vectorMemoryAgent()
    await ag.runBasicTest()

    ag.vector_memory.reset()
    print("Memory cleared")

    await ag.userSetupVectorDatabase()
    print("Messages inputted")

    await ag.userTests()
    print("Tests done")

if __name__ == "__main__":
    asyncio.run(main())

    #ollama tests
    # ol = vectorMemoryAgent(
    #     llm = Ollama(model="qwen3:4b")
    # )
    # ol.runBasicTest()

