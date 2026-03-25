from flask import Flask, render_template, request
from paraphraser import paraphrase
from sentiment import analyze_sentiment
from features import analyze_features
from ner import extract_entities
from topics import analyze_topics

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    text       = ""
    active_tab = "paraphrase"
    results    = {}

    if request.method == 'POST':
        text       = request.form.get('text', '').strip()
        active_tab = request.form.get('active_tab', 'paraphrase')

        if text:
            if active_tab == 'paraphrase':
                tone = request.form.get('tone', 'neutral')
                results['paraphrase'] = {
                    'output': paraphrase(text, tone),
                    'tone': tone,
                }

            elif active_tab == 'sentiment':
                results['sentiment'] = analyze_sentiment(text)

            elif active_tab == 'features':
                results['features'] = analyze_features(text)

            elif active_tab == 'ner':
                results['ner'] = extract_entities(text)

            elif active_tab == 'topics':
                results['topics'] = analyze_topics(text)

    return render_template('index.html',
                           text=text,
                           active_tab=active_tab,
                           results=results)

if __name__ == '__main__':
    app.run(debug=True)