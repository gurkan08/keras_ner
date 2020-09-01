# keras bi-directional lstm + crf ner service

docker build -t keras_ner .

docker run -it -p 8000:8000 keras_ner

dataset: https://groups.csail.mit.edu/sls/downloads/movie/

# test samples
text = "show me films with drew barrymore from the 1980s"
text = "who directed the film pulp fiction that starred john travolta"
text = "which film has the highest viewer rating this year"
text = "what was the first movie in color"
text = "who diected the first james bond movies"
text = "list childrens movies with billy crystal"
text = "gürkan şahin ytü bilgisayar mühendisliği"

# example request
API endpoint: http://localhost:8000/ner/

text: "which film has the highest viewer rating this year"

{
    "result": "o o o o b-ratings_average i-ratings_average i-ratings_average b-year i-year"
}
