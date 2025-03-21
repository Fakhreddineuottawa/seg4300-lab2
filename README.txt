Sentiment Analysis API using Flask and a pretrained DistilBERT model, containerized with Docker.
Instructions to Run Locally:
1.docker pull levibash/sentiment-analysis:latest
2.docker run -p 5000:5000 levibash/sentiment-analysis
3.curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d "{\"text\": \"Hello world!\"}"

Docker Hub Link:
https://hub.docker.com/r/levibash/sentiment-analysis

Fakhreddine Bouaziz	
300210384