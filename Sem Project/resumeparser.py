import time
import requests
import openai
from openai import OpenAI, Client
import yaml

api_key = None
CONFIG_PATH = r"config.yaml"

with open(CONFIG_PATH) as file:
    data = yaml.load(file, Loader = yaml.FullLoader)
    api_key = data['OPEN_API_KEY']

def ats_extractor(resume_data,max_retries=3,delay = 5):

    prompt = '''
    You are an AI bot designed to act as a professional for parsing resumes.Yor are given with resume and your job is to extract the following information from the resume:
    1.Full name
    2.Email Id
    3.Github portfolio
    4.Linkedin profile
    5.Employment details
    6.Technical skills
    7.Soft skills
    Give the exracted information in json format only
    '''

    openai_client = OpenAI(
        api_key = api_key,
    )
    messages = [
        {"role": "system",
         "content": prompt}
    ]
    user_content = resume_data

    messages.append({"role": "user","content": user_content})

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                temperature = 0.0,
                max_tokens = 1500
            )
            if response and response.choices:
                data = response.choices[0].message.content
                return data
            else:
                print(f"API response is empty or invalid on attempt {attempt+1}")
        except requests.exceptions.HTTPError as e:
            print(f"Rate limit exceeded: {e}.Retrying in {delay} seconds")
            time.sleep(delay)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(delay)
        return None