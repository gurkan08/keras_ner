# keras bi-directional lstm + crf ner service

docker build -t keras_ner .

docker run -it -p 8000:8000 keras_ner

# example request
API endpoint: http://localhost:8000/ner/

text: which film has the highest viewer rating this year

{
    "result": "o o o o b-ratings_average i-ratings_average i-ratings_average b-year i-year"
}
