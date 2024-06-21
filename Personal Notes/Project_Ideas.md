1. An agent that can write code to generate other agents based on the user request.
    - The agent creating agent will test the generated agent based on the requirements set by the user.
    - The agent will let the user test the generated agent by passing along the user input
2. Meeting transcription/analysis/recall tool
    - Voice identification (see https://huggingface.co/pyannote/segmentation-3.0 or similar)
        - Use a key phrase like "Hello {App Name}, I'm {Speaker Name}"
        - Use speech to text to identify name that can be tied to the voice profile
        - Find a way to store this in a database for recall
        - Unidentified voices will store an accessible audio clip for manual override of identification (user selects voice profile and enters a name)
    - Speech to text transcription/storage with identified voice (see https://huggingface.co/openai/whisper-large-v3 or similar)
        - Probably store this in a traditional database with the following columns:
            - transcribed text
        - 