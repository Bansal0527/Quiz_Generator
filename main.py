import json
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.callbacks import get_openai_callback
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint



from dotenv import load_dotenv
import pandas as pd
import traceback
from util import parse_file, RESPONSE_JSON, get_table_data

load_dotenv()
# llm = OpenAI(model_name="text-davinci-003")  # Use the appropriate model name
# llm = ChatGoogleGenerativeAI(model="gemini-pro")
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
)


from langchain.llms import OpenAIChat
# llm = OpenAIChat(model_name="gpt-3.5-turbo")

template1 = """
Text:{text}
You are an expert MCQ maker, Given the above text, it is your job to \n
create a quiz of {number} multiple choice questions for grade {grade} student in {tone} tone,
Make sure that the questions are not repeated and check all the questions to be conforming the text as well
Make sure to format your response like RESPONSE_JSON below and use it as a guide.\
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""
quiz_generator_prompt = PromptTemplate(
    input_variables=["text", "number", "grade", "tone", "response_json"],
    template=template1,
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generator_prompt, output_key='quiz', verbose=True)

template = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {grade} for grade students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz if the students will be 
able to understand the questions and answer them. Only use at max 50 words for complexity analysis.
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.
Quiz_MCQs:
{quiz}

Critique from an expert English Writer of the above text
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=["grade", "quiz"], template=template)
review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key='review', verbose=True)

# Overall chain
generate_evaluate_chain = SequentialChain(chains=[quiz_chain, review_chain],
                                                input_variables=["text", "number", "grade", "tone", "response_json"],
                                                output_variables=["quiz", "review"], verbose=True)

st.title("MCQs Generator Application")

with st.form("user_input"):
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF or a txt file")

    # Input Fields
    mcq_count = st.number_input("No of MCQs", min_value=3, max_value=50)

    # Grade
    grade = st.number_input("Insert Grade", min_value=1, max_value=10)
    # Quiz Tone
    tone = st.text_input("Insert Quiz tone", max_chars=100, placeholder="simple")

    # Add button
    button = st.form_submit_button("Create MCQs")
    # Check if button is clicked
    if button and uploaded_file is not None and mcq_count and grade and tone:
        with st.spinner("Loading..."):
            try:
                text = parse_file(uploaded_file)
                # Count Tokens
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "grade": grade,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")

                if isinstance(response, dict):
                    # Extract the quiz data from the response
                    quiz = response.get("quiz", None)

                    if quiz is not None:
                        table_data = get_table_data(quiz)

                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)

                            # Display the review in a text box as well
                            st.text_area(label="Review", value=response["review"])

                        else:
                            st.error("Error in the table data")

                else:
                    st.write(response)
