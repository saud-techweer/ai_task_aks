1. Install Ollama: brew install -cask ollama
2. Install Deepseek: ollama run deepseek-r1:1.5b
3. Install Docker: brew install --cask docker
4. Install open-webui using docker: docker run -d -p 3000:8080 -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama

Connecting Open WebUI to our model:
1. Go to setting
2. Select admin setting
3. Click on connection here you will see ollama section here add this http://host.docker.internal:11434
