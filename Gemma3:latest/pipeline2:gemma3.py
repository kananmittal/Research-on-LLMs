
import pdfplumber
from ollama import Client
import re

import os
from openai import OpenAI

ollama_client = Client(host="http://127.0.0.1:11434")
GEMMA_MODEL = "gemma3:latest"
client = OpenAI(api_key="API KEY")


def extract_clean_transcript(pdf_path, start_page=1, end_page=13):
    transcript = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[start_page:end_page]:
            text = page.extract_text()
            if text:
                transcript += text + "\n"
    return transcript

pdf_path = "Transcript.pdf"  
clean_transcript = extract_clean_transcript(pdf_path, start_page=1, end_page=13)  

pdf_path = "Notes.pdf"  
extract_notes = (pdf_path,)  

def ask_gemma(clean_transcript: str, extract_notes: str ) -> str:
    prompt = f"""
Let's generate a consolidated summary of the two source documents: a transcript of an earnings call (conference call) and a note (bullet form) derived from the same transcript
1) Read through the entire transcript and notes carefully to understand the context.
2) Identify and extract the key topics and insights discussed in depth from the documents.
3) Pay attention to any numerical data presented in the documents.
4) When including numbers in the summary, ensure they are:
	a) Explicitly stated values from the documents (do not fabricate numbers).
	b) Appropriately represented with clear context from the documents.
5) Synthesize the extracted information and numbers into a concise summary
 that flows logically.
6) Conserve all the important information from the transcript and notes, so that you can use it to answer any question from it.
Documents:
=== Transcript ===
{clean_transcript, extract_notes}
"""

    response = ollama_client.chat(
        model=GEMMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Usage:
Summary_text= ask_gemma(clean_transcript, extract_notes )
print(Summary_text)

QF = [
    "What was the total revenue of TCS for the financial year 2025 in dollar terms?",
    "What was TCS's net profit margin for FY 2025?",
    "Which TCS product recorded the highest number of new wins and go-lives in Q4 FY25?",
    "How many clients contributed over $100 million in revenue during FY 2025?",
    "What was the total deal TCV in Q4 FY25, and how much of it came from North America?"
]

QA = [
    "How did TCS manage to maintain margins despite headwinds from wage increases and infrastructure investments?",
    "Why is TCS's focus on GenAI and AI-for-business expected to provide long-term strategic advantage?",
    "How might the geopolitical and macroeconomic uncertainties affect TCS's short-term and long-term growth strategy?",
    "What can be inferred about TCS's HR and skilling strategy based on their campus hiring and AI skilling initiatives?",
    "Considering the growth and TCV figures, do you think TCS's cautious optimism for FY26 is justified? Why or why not?"
]


def ask_gemma(Summary_text: str, question: str) -> str:
    prompt = (
        "You are an expert analyst.  Answer the given questions using  the summary text created by you in one line. All the information is therre in it "

        f"DOCUMENT:\n{Summary_text}\n\nQUESTION:\n{question}"
    )
    resp = ollama_client.chat(
        model=GEMMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a factual document QA bot."},
            {"role": "user",   "content": prompt}
        ],

    )
    return resp["message"]["content"]

def score_response_openai(answer: str, question: str, mode: str = "hallucination") -> float:
    """
    mode="hallucination": 0 = fully factual, 100 = fully hallucinated
    mode="relevance":     0 = completely irrelevant, 100 = fully relevant
    """
    if mode == "hallucination":
        score_prompt = (
            "You are a hallucination detection specialist. On a scale from 0 to 100, "
            "where 0 means fully factual and 100 means fully hallucinated, rate the following "
            "answers to the questions. Only reply with the number.\n\n"
            f"QUESTION: {question}\n\nANSWER: {answer}"
        )
    else:
        score_prompt = (
            "You are a relevance evaluator. On a scale from 0 to 100, where 0 means "
            "completely irrelevant and 100 means fully relevant, rate the following answers "
            "to the questions. Only reply with the number.\n\n"
            f"QUESTION: {question}\n\nANSWER: {answer}"
        )
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # GPT-4 API model
            messages=[
                {"role": "system", "content": "You evaluate factual consistency."},
                {"role": "user", "content": score_prompt}
            ],
            temperature=0.0,
        )
        score_text = response.choices[0].message.content.strip()
        m = re.search(r"(\d+(\.\d+)?)", score_text)
        return float(m.group(1)) if m else None
    except Exception as e:
        print(f"Error scoring with OpenAI: {e}")
        return None
for q in QF:
    ans_unique = ask_gemma(Summary_text, q)
    score_uni = score_response_openai(ans_unique, q, mode="hallucination")
    print(f"\nQUESTION: {q} \nAnswer {ans_unique} ")
    print("=====================================================================================================================================================")
    print(f" • Summary_text → Hallucination score:     {score_uni}")

for q in QA:
    ans_unique = ask_gemma(Summary_text, q)
    score_uni = score_response_openai(ans_unique, q, mode="hallucination")
    print(f"\nQUESTION: {q} \nAnswer {ans_unique} ")
    print("=====================================================================================================================================================")
    print(f" • Summary_text → Hallucination score:     {score_uni}")