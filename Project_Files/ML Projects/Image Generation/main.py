
from flask import Flask,render_template, request, send_file
import openai
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
openai.api_key = "API-KEY"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    prompt = request.form['prompt']
    image = generate_image(prompt)

    image.save('static/generated_image.jpg')
    return render_template('result.html',image_url='static/generated_image.jpg')

def generate_image(prompt):
    response = openai.Image.create(
        prompt = prompt,
        n=1,
        size = "1024x1024"
    )
    image_url = response['data'][0]['url']
    image_response = requests.get(image_url)
    img = Image.open(BytesIO(image_response.content))
    return img

if __name__ == '__main__':
    app.run(debug=True)
