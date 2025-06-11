**Objectives**

Healthcare providers increasingly seek AI-assisted tools to offer preliminary guidance and triage for common medical queries. This project transforms a decoder-only transformer model into a medically informed chatbot capable of:
Responding to medical questions using professional language.
Adhering to clinical guidelines.
Providing non-substitutive disclaimers.
Ensuring HIPAA-equivalent anonymization of all inputs.


**Tech Stack**

Language & Libraries: Python, PyTorch, Hugging Face Transformers
Fine-Tuning Tools: PEFT, BitsAndBytes
Interface: Streamlit

**SetUp**

Ensure you have Pipenv installed. You can install it using:
pip install pipenv

Then, run the following commands to set up the environment:

pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf

pipenv install huggingface_hub

pipenv install streamlit

**To run the streamlit interface**

streamlit run chatbot_interface.py
