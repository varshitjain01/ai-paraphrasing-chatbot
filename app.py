from flask import Flask, render_template, request
from paraphraser import paraphrase

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    user_text = ""

    if request.method == 'POST':
        user_text = request.form['text']
        result = paraphrase(user_text)

    return render_template('index.html', result=result, user_text=user_text)

if __name__ == '__main__':
    app.run(debug=True)