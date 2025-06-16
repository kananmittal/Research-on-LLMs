#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pdfplumber
from ollama import Client
import re

def extract_clean_transcript(pdf_path, start_page=1):
    transcript = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[start_page:]:
            text = page.extract_text()
            if text:
                transcript += text + "\n"
    return transcript

pdf_path = "tcs_transcript1.pdf"  
clean_transcript = extract_clean_transcript(pdf_path, start_page=1)  

ollama_client = Client(host="http://127.0.0.1:11434")
DEEPSEEK_MODEL = "deepseek-r1:7b"
def ask_deepseek(clean_transcript: str ) -> str:
    prompt = f"""
You are given a document i.e. Transcript.

Your task is to create a summary of the Transcript.
STEPS:
- Carefully read the Transcript.
- Extract the  meaningful, and relevant insights from the Transcript.
- Ensure the summary document reads like notes that are complete, natural, and coherent write-up 
- Preferably use bullet points.
- Include all the data, figures, events, and key commentary 
- Do not include the Question and Answer round in the summary.

Documents:
=== Transcript ===
{clean_transcript}
"""

    response = ollama_client.chat(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Usage:
Summary_text= ask_deepseek(clean_transcript)
print(Summary_text)


# In[4]:


def ask_deepseek(Summary_text: str, clean_transcript: str) -> str:
    prompt = (
        "You are given two documents i.e. clean_transcript and Summary. summary is created by you, act as a self critique to chceck for missing information. If there is any missing information, add it in the summary text. "
       
        f"DOCUMENT:\n{Summary_text}\n\nDOCUMENT:\n{clean_transcript}"
    )
    response = ollama_client.chat(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
Summary_text2= ask_deepseek(Summary_text, clean_transcript)
print(Summary_text2)


# In[6]:


QF = [
    "What was the total revenue of TCS for the financial year 2025 in dollar terms?",
    "What was TCS’s net profit margin for FY 2025?",
    "Which TCS product recorded the highest number of new wins and go-lives in Q4 FY25?",
    "How many clients contributed over $100 million in revenue during FY 2025?",
    "What was the total deal TCV in Q4 FY25, and how much of it came from North America?"
]

QA = [
    "How did TCS manage to maintain margins despite headwinds from wage increases and infrastructure investments?",
    "Why is TCS’s focus on GenAI and AI-for-business expected to provide long-term strategic advantage?",
    "How might the geopolitical and macroeconomic uncertainties affect TCS’s short-term and long-term growth strategy?",
    "What can be inferred about TCS’s HR and skilling strategy based on their campus hiring and AI skilling initiatives?",
    "Considering the growth and TCV figures, do you think TCS’s cautious optimism for FY26 is justified? Why or why not?"
]



from ollama import Client
import re
# create a client bound to your Docker container
ollama_client = Client(host="http://127.0.0.1:11434")

def ask_deepseek(Summary_text2: str, question: str) -> str:
    prompt = (
        "You are an expert analyst.  Answer the given questions using  the new summary text created by you in one line. All the information is therre in it "
       
        f"DOCUMENT:\n{Summary_text2}\n\nQUESTION:\n{question}"
    )
    resp = ollama_client.chat(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "You are a factual document QA bot."},
            {"role": "user",   "content": prompt}
        ],

    )
    return resp["message"]["content"]


# In[10]:


LLAMA_MODEL = "llama3"
def score_response(LLAMA_MODEL: str, answer: str, question: str, mode: str = "hallucination") -> float:
    """
    mode="hallucination": 0 = fully factual, 100 = fully hallucinated
    mode="relevance":     0 = completely irrelevant, 100 = fully relevant
    """
    if mode == "hallucination":
        score_prompt = (
            "You are a hallucination detection specialist.  On a scale from 0 to 100, "
            "where 0 means fully factual and 100 means fully hallucinated, rate the following "
            "answer to the question.  Only reply with the number.\n\n"
            f"QUESTION: {question}\n\nANSWER: {answer}"
        )
    else:
        score_prompt = (
            "You are a relevance evaluator.  On a scale from 0 to 100, where 0 means "
            "completely irrelevant and 100 means fully relevant, rate the following answer "
            "to the question.  Only reply with the number.\n\n"
            f"QUESTION: {question}\n\nANSWER: {answer}"
        )
    client = ollama_client
    res = ollama_client.chat(
       
        model="llama3",
        messages=[
            {"role": "system", "content": "You evaluate factual consistency."},
            {"role": "user",   "content": score_prompt}
             ],
    )
    text = res.message.content.strip()
    m = re.search(r"(\d+(\.\d+)?)", text)
    return float(m.group(1)) if m else None


# In[11]:


for q in QF:
    ans_unique = ask_deepseek(Summary_text2, q)
    score_uni = score_response(LLAMA_MODEL, ans_unique, q, mode="hallucination")


    print(f"\nQUESTION: {q} \nAnswer {ans_unique} ")
    print("=====================================================================================================================================================")
    print(f" • Summary_text → Hallucination score:     {score_uni}")


# In[12]:


for q in QA:
    ans_unique = ask_deepseek(Summary_text2, q)
    score_uni = score_response(LLAMA_MODEL, ans_unique, q, mode="hallucination")


    print(f"\nQUESTION: {q} \nAnswer {ans_unique} ")
    print("=====================================================================================================================================================")
    print(f" • Summary_text → Hallucination score:     {score_uni}")

