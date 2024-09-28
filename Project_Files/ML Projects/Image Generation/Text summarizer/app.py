from flask import Flask, render_template, request
from summarizer import TextSummarizer

app = Flask(__name__)

# Create an instance of the TextSummarizer
summarizer = TextSummarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        # Generate the summary
        summary = summarizer.summarize(text)
        return render_template('result.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
