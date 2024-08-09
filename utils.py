import pandas as pd
import os
from groq import Groq
import time 
from langchain_chroma import Chroma
from tqdm import tqdm
import os
from langchain_community.embeddings import JinaEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import initialize_agent
from langchain_groq import ChatGroq
import json
import shutil
import glob


with open('keys.json', 'r') as file:
    api_keys = json.load(file)

os.environ["GROQ_API_KEY"] = api_keys["GROQ_API_KEY"]
os.environ["JINA_API_KEY"] = api_keys["JINA_API_KEY"]
os.environ["GOOGLE_CSE_ID"] = api_keys["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = api_keys["GOOGLE_API_KEY"]

def call_groq(raw_prompt, temperature=0):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    attempt = 0
    while attempt < 5:
        try:
            chat_completion = client.chat.completions.create(
                temperature=temperature,
                max_tokens=8192,
                messages=[
                    {
                        "role": "system",
                        "content": "you are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": raw_prompt,
                    }
                ],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except:
            print("Rate limite exceeded, sleeping for 60 seconds")
            time.sleep(60)
            attempt += 1
    print("Failed to generate!")
    return None

# These functions use for getting relevant textfor a query
class CustomJinaEmbeddings:
    def __init__(self):
        self.embedding_func = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embedding_func = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
        batch_size = 2048
        embeddings = []
        n = len(texts)
        for i in tqdm(range(0, n, batch_size)):
            page_contents = []
            for j in range(i, i + batch_size):
                if j >= n:
                    break
                page_contents.append(texts[j])
            if len(page_contents) > 0:
                embeddings.extend(embedding_func.embed_documents(page_contents))
        return embeddings 
    def embed_query(self, text: str) -> List[float]:
        return self.embedding_func.embed_query(text)

def create_groq_evaluate_prompt(query, context):
    prompt = f"Here is a context I want you to consider: {context}\n"
    prompt += f"Here is a query: {query}\n" 
    prompt += """Write a brief analysis of whether the context is related to the query. Then conclude by writing "Result:", followed by "yes" or "no"."""
    return prompt

def parse_res(header, response):
    try:
        res = response[response.index(header) + len(header):].strip()
        return res 
    except:
        return None

def evaluate_context(query, context):
    evaluate_prompt = create_groq_evaluate_prompt(query, context)
    result = parse_res("Result:", call_groq(evaluate_prompt))
    if result == "yes":
        return True 
    else:
        return False

def get_texts_by_query(query, num_texts=100, retriever=None):
    if retriever == None:
        embeddings = CustomJinaEmbeddings()
        vectordb = Chroma(persist_directory="fineweb_db_new", embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": num_texts})
    description = call_groq(f"Elaborate on this: {query}.")
    docs = retriever.get_relevant_documents(description)
    texts = [doc.page_content for doc in docs]
    evaluate_result = []
    for i in range(len(texts)):
        evaluate_result.append(evaluate_context(query, texts[i]))
    remaining_texts = []
    for i in range(len(evaluate_result)):
        if evaluate_result[i] == True:
            remaining_texts.append(texts[i])
    return remaining_texts


# These functions are used for genq process
def create_genq_prompt(context, n1=20, n2=20):
    prompt = "Here is the context you need to consider: "
    prompt += f"{context} \n"
    prompt += f"Now, list {n1} topics that you can answer questions about in relation to this context. Select a random topic from this list and specify it.\n"
    prompt += f"Then write {n2} subtopics about the selected topic. Select a random subtopic from this list and specify it.\n"
    prompt += "Next, write a question that is not directly related to the subtopic but requires expertise in the subtopic and the given context."
    prompt += "The name of the subtopic should not appear in the question, and the words in the subtopic should not be used in the question."
    prompt += """Start your questions with "Question:". Be creative."""
    return prompt 

def parse_res(header, response):
    try:
        res = response[response.index(header) + len(header):].strip()
        return res 
    except:
        return None

def genq_by_context(context, n1=20, n2=20, max_attempt=5):
    genq_prompt = create_genq_prompt(context, n1, n2)
    attempt = 0
    while attempt < max_attempt:
        response = call_groq(genq_prompt, temperature=0.8)
        genq_result = parse_res("Question:", response)
        if genq_result != None:
            return genq_result 
        attempt += 1
    return None 

def gen_m_q_for_n_context(contexts, m, n1=20, n2=20, max_attempt=5):
    all_q = pd.DataFrame(columns=["text", "instruction"])
    for i in range(len(contexts)):
        print(f"{i}/{len(contexts)} processed")
        for _ in range(m):
            try:
                result = genq_by_context(contexts[i], n1, n2, max_attempt)
                if result != None:
                    all_q.loc[len(all_q)] = [contexts[i], result]
            except:
                print(f"Failed to generate for {i}-th context! Skipping it...")
                break
    return all_q


# These functions are use to use Groq llm as agent to search google
class CustomGoogleSearchWrapper:
    def __init__(self, k):
        self.all_resources = []
        self.search = GoogleSearchAPIWrapper()
        self.k = k
    def top_k_results(self, query):
        res = self.search.results(query, self.k)
        results = " ".join([r["snippet"] for r in res])
        res = [r for r in res if "snippet" in r]
        sources = [r["link"] for r in res]
        self.all_resources.extend(sources)
        return {"search_results" : results, "sources" : sources}
    def get_all_resources(self):
        return self.all_resources
    def reset_all_resources(self):
        self.all_resources = []

def search_for_query(query, llm=None, k=10):
    if llm == None:
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", max_tokens=8192)
    search_engine = CustomGoogleSearchWrapper(k)
    meta_data_search_tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=search_engine.top_k_results,
    )
    tools = [meta_data_search_tool]
    agent = initialize_agent(
        tools,
        llm,
        agent="chat-zero-shot-react-description",
        verbose=True,
        agent_kwargs={
            "max_execution_time": 3000,
            "llm_prefix": (
                "Provide a comprehensive analysis of the information gathered, covering all relevant aspects in depth. Then, respond in detail to the query."
            )
        }
    )
    attempt = 0
    while attempt < 5:
        try:
            res = agent.run(f"{query}. Please provide multiple perspectives and a detailed breakdown.")
            sources = search_engine.get_all_resources()
            # search_engine.reset_all_resources()
            return res, sources 
        except:
            print("Failed to generate, sleeping for 60 seconds")
            time.sleep(60)
            attempt += 1
    return "", []

def search_for_all_queries(instruction_df, original_query):
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", max_tokens=8192)
    completed_df = pd.DataFrame(columns = ["original_context", "instruction", "response", "sources", "original_query"])
    all_contexts = list(instruction_df["text"])
    all_instructions = list(instruction_df["instruction"])
    for i in tqdm(range(len(instruction_df))):
        res, sources = search_for_query(all_instructions[i], llm)
        if res != None and len(res) >= 80 and "action_input" not in res:
            # Format results
            if "The final answer to the question is:" in res and res.index("The final answer to the question is:") == 0:
                res = res[len("The final answer to the question is:"):].strip()
            if "The final answer to the original input question is:" in res and res.index("The final answer to the original input question is:") == 0:
                res = res[len("The final answer to the original input question is:"):].strip()
            if "The final answer to the original input question." in res and res.index("The final answer to the original input question.") == 0:
                res = res[len("The final answer to the original input question."):].strip()
            completed_df.loc[len(completed_df) + 1] = [all_contexts[i], all_instructions[i], res, sources, original_query]
    return completed_df

# some utils
def delete_all_contents(folder_path):
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  
    print(f"All contents of the folder {folder_path} have been deleted.")

def delete_files_with_prefix(folder_path, prefix):
    pattern = os.path.join(folder_path, f"{prefix}*")
    files_to_delete = glob.glob(pattern)
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")