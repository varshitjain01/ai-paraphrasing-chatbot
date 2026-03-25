from flask import Flask, render_template, request
from paraphraser import paraphrase

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    user_text = ""
    tone = "neutral"

    if request.method == 'POST':
        user_text = request.form.get('text', '')
        tone = request.form.get('tone', 'neutral')
        result = paraphrase(user_text, tone)

    return render_template('index.html', result=result, user_text=user_text, tone=tone)

if __name__ == '__main__':
    app.run(debug=True)