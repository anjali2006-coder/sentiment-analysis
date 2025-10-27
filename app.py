from flask import Flask,render_template , request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
app = Flask(__name__)


# loading vectorizer and model
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
model= joblib.load('model/model.pkl')

# nltk.download('stopwords')

#text preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


@app.route('/')
def hello_world():
   return render_template('index.html')
#    return "hello_world"

@app.route('/predict', methods=['POST'])
def predict():
   if request.method == 'POST':
      user_text = request.form['text']
      #preprocessing
      cleaned_text = preprocess_text(user_text)
      #vectorize
      transformed_text = vectorizer.transform([cleaned_text])
      #text sentiment analyze
      prediction = model.predict(transformed_text)[0]
      #the model will return the output in array form array(['positive'], dtype=object)
      #so to get single string output [0] is used
      sentiment = prediction.capitalize()
      return render_template('index.html', text=user_text, sentiment=sentiment)


if __name__ == "__main__":
   app.run(debug=True)