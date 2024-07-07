# Course Final Project Ideas

## Agent Generating Agents
An agent that can write code to generate other agents based on the user request.
- The agent creating agent will test the generated agent based on the requirements set by the user.
- The agent will let the user test the generated agent by passing along the user input

## Meeting Note Integrated RAG
- Voice detection to begin recording for transcription.
- Voice identification (see https://huggingface.co/pyannote/segmentation-3.0, https://huggingface.co/pyannote/speaker-diarization-3.0, or similar)
    - Use a key phrase like "Hello {App Name}, I'm {Speaker Name}"
    - Use speech to text to identify name that can be tied to the voice profile
    - Find a way to store this in a database for recall
    - Unidentified voices will store an accessible audio clip for manual override of identification (user selects voice profile and enters a name)
- Speech to text transcription/storage with identified voice (see https://huggingface.co/openai/whisper-large-v3 or similar)
    - Probably store this in a traditional database with the following columns:
        - Meeting name
        - privilege level
        - speaker id
        - timestamp of audio clip start
        - duration of clip
        - Maybe the audio file itself?
        - The vector embedding of the text transcription (for RAG)
- RAG pipeline to ingest transcriptions and respond to questions
    - Answer verbal and text questions 
    - Question answering requires the following:
        - Cosine similiarity on all vector embeddings
        - After identifying the the vectors, search for them in the DB
        - Feed the rest of the context in the DB to the model as well (Who was the speaker, when was it said, what meeting it is from)
        - Stream the text response (maybe do TTS too)
        - Store questions and answers in a DB

## Image to 3D model (GPT3D) 
- Use a depth mapping model and gpt-4o to generate a point cloud that can be converted to a 3D model and 3D printed.
     - https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
- Load into solidworks and run featureworks on it


## API Assistant
- The LLM will search the internet for API documentation for a given 

### Links
1. https://docs.chainlit.io/advanced-features/multi-modal (multimodal chainlit integration)