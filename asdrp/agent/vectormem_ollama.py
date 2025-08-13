'''
Followed https://ollama.com/blog/embedding-models as reference
Working on making more customizable for testing
'''

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import ollama
import chromadb

k = 2 # default amount of relevant to show up
msgs = []
userInput = input("Enter a fact: ")
while userInput:
    msgs.append(userInput)
    userInput = input("Enter a fact: ")

cli = chromadb.Client()
coll = cli.create_collection(name = "docs")

for i in range(len(msgs)):
    resp = ollama.embed(model='mxbai-embed-large', input = msgs[i])
    emb = resp["embeddings"]
    coll.add(ids=[str(i)], embeddings=emb, documents=[msgs[i]])

test = input("Test: ")
resp = ollama.embed(
    model="mxbai-embed-large",
    input=test
)
results = coll.query(
    query_embeddings=resp["embeddings"],
    n_results=k
)

# output gathered data
data = ""
for i in results['documents'][0]:
    print(i)
    data += i + "\n"

#get response
op = ollama.generate(
    model="qwen3:4b",
    prompt=f"Previous data: {data}\nRespond to this: {test}"
)

print(op['response'])