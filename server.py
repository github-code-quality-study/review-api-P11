import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    # Define allowed locations to be used with GET and POST
    allowed_locations = [
        "Albuquerque, New Mexico",
        "Carlsbad, California",
        "Chula Vista, California",  
        "Colorado Springs, Colorado",
        "Denver, Colorado",
        "El Cajon, California",
        "El Paso, Texas",
        "Escondido, California",
        "Fresno, California",
        "La Mesa, California",
        "Las Vegas, Nevada",
        "Los Angeles, California",
        "Oceanside, California",
        "Phoenix, Arizona",
        "Sacramento, California",
        "Salt Lake City, Utah",
        "San Diego, California",
        "Tucson, Arizona"
    ]

    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Write your code here
            query_string = environ.get('QUERY_STRING', '')
            query_params = parse_qs(query_string)
            location_param = query_params.get('location', [None])[0]
            if location_param:
                location_param = location_param.replace('+', ' ')

            start_date_param = query_params.get('start_date', [None])[0]
            if start_date_param:
                start_date_param = datetime.strptime(start_date_param, '%Y-%m-%d').date()

            end_date_param = query_params.get('end_date', [None])[0]
            if end_date_param:
                end_date_param = datetime.strptime(end_date_param, '%Y-%m-%d').date()

            filtered_reviews = [
            review for review in reviews
            if review["Location"] in self.allowed_locations and
            (not location_param or review["Location"] == location_param) and
            (not start_date_param or datetime.strptime(review["Timestamp"], '%Y-%m-%d %H:%M:%S').date() >= start_date_param) and
            (not end_date_param or datetime.strptime(review["Timestamp"], '%Y-%m-%d %H:%M:%S').date() <= end_date_param)
        ]

            # Analyze sentiment for each review in the response body
            sentiments = []
            for review in filtered_reviews:
                sentiment_scores = self.analyze_sentiment(review["ReviewBody"])
                sentiments.append(sentiment_scores)

            # Add the sentiment scores to the reviews
            for i, review in enumerate(filtered_reviews):
                review["sentiment"] = sentiments[i]

            filtered_reviews.sort(key=lambda x: x["sentiment"]["compound"], reverse=True)

            # Convert the reviews to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length)
                post_data = parse_qs(post_data.decode('utf-8'))

                review_body = post_data.get('ReviewBody', [None])[0]
                location = post_data.get('Location', [None])[0]

                if not review_body or not location or not location in self.allowed_locations:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    response_body = json.dumps({"error": "ReviewBody and Location are required and must be valid."}).encode("utf-8")
                    return [response_body]

                review = {
                    "ReviewId": str(uuid.uuid4()),
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Location": location,
                    "ReviewBody": review_body
                }

                # Append the new review to the reviews list
                reviews.append(review)

                start_response("201 Created", [
                    ("Content-Type", "application/json")
                ])

                # Dump the response body
                response_body = json.dumps({
                    "ReviewId": review["ReviewId"],
                    "ReviewBody": review["ReviewBody"],
                    "Location": review["Location"],
                    "Timestamp": review["Timestamp"]
                }).encode("utf-8")
                
                return [response_body]

            except Exception as e:
                start_response("400 Bad Request", [
                  ("Content-Type", "application/json")
                ])
                response_body = json.dumps({"error": str(e)}).encode("utf-8")
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()