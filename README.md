# Gemini-Image-Library-Agent

This project illustrates the beginnings of a basic digital asset management system that is powered by Google's Gemini API. The system is designed to support management of photography image libraries. The project current explores importing images into a database, capturing and generating descriptive metadata, and searching the image library–all with the help of a Gemini chat agent.

## Usage

The chat interface is run using streamlit. Enter the command below in your cli to launch the server in your web browser.

```commandline
streamlit run gemini_image_library_agent.py

```

## Google Gen AI SDK (Gemini API)

Google Gen AI Python SDK provides an interface for developers to integrate Google's generative models into their Python applications.

Documentation: [https://googleapis.github.io/python-genai/](https://googleapis.github.io/python-genai/)

pypi: [https://pypi.org/project/google-genai/](https://pypi.org/project/google-genai/)

### API Key

Keep your Gemini API in streamlit secrets.

Add a line to the secrets.toml file: GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

```commandline
.streamlit/secrets.toml

```


