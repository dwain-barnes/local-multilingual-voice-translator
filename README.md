# Real-Time Multilingual Voice Translator

This project is a real-time, multilingual voice translator that leverages the power of local AI models for speech-to-text, translation, and text-to-speech. It is designed to be a powerful and flexible tool for anyone who needs to communicate across language barriers.

## Key Features

-   **Real-Time Translation**: Speak in any supported language and hear translations with natural voice synthesis.
-   **Multilingual Support**: Supports over 23 languages for both translation and voice synthesis.
-   **Local AI-Powered**: Utilizes local models, ensuring privacy and offline functionality.
-   **High-Quality Voice Synthesis**: Powered by Chatterbox TTS for natural-sounding voice output.
-   **Accurate Speech-to-Text**: Integrates with Distil-Whisper FastRTC for precise transcriptions.
-   **Web Interface**: User-friendly web interface built with Gradio and FastAPI.

## Supported Languages

| Code | Language   |
|:-----|:-----------|
| ar   | Arabic     |
| da   | Danish     |
| de   | German     |
| el   | Greek      |
| en   | English    |
| es   | Spanish    |
| fi   | Finnish    |
| fr   | French     |
| he   | Hebrew     |
| hi   | Hindi      |
| it   | Italian    |
| ja   | Japanese   |
| ko   | Korean     |
| ms   | Malay      |
| nl   | Dutch      |
| no   | Norwegian  |
| pl   | Polish     |
| pt   | Portuguese |
| ru   | Russian    |
| sv   | Swedish    |
| sw   | Swahili    |
| tr   | Turkish    |
| zh   | Chinese    |

## Installation Guide

Follow these steps precisely to ensure all dependencies are installed in the correct order.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/dwain-barnes/multilingual-voice-translator-realtime.git
    cd multilingual-voice-translator-realtime
    ```

2.  **Create and Activate a Conda Environment**
    
    We recommend using Anaconda to manage the environment, as shown in the setup video.
    ```bash
    # Create a new environment named 'translator' with Python 3.11
    conda create -n translator python=3.11 -y

    # Activate the new environment
    conda activate translator
    ```

3.  **Install Dependencies in Order**

    Run each of the following commands one by one. This specific order is crucial for the application to work correctly.
    ```bash
    # 1. Install core numerical and scientific libraries
    pip install numpy scipy

    # 2. Install the specific PyTorch version required by Chatterbox (with CUDA 11.8)
    # If you don't have an NVIDIA GPU, you can try removing "+cu118" and the --index-url
    pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118

    # 3. Install other requirements for TTS and audio processing
    pip install librosa transformers diffusers safetensors requests httpx pyaudio

    # 4. Install Chatterbox TTS
    pip install chatterbox-tts

    # 5. Install dependencies for the web server, real-time communication, and STT
    pip install python-dotenv
    pip install "fastrtc[vad,stt,tts]"
    pip install distil-whisper-fastrtc
    pip install openai
    pip install uvicorn
    ```

4.  **Set Up Local AI Models**
    -   **LM Studio**: Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/).
        -   After installing, search for and download the **`Hunyuan-MT-7B-GGUF`** model from the in-app model browser.
        -   Once downloaded, navigate to the local server tab (`<->`) and load the model.
        -   Start the server.
    -   **Chatterbox & Whisper Models**: The required models for text-to-speech and speech-to-text will be downloaded automatically the first time you run the application.

5.  **Create Environment File**

    Create a file named `.env` in the root directory (`local-multilingual-voice-translator`) and add the following content:
    ```    LM_STUDIO_BASE_URL=http://localhost:1234/v1
    LM_STUDIO_API_KEY=lm-studio
    WHISPER_MODEL=distil-whisper/distil-large-v3
    ```

## How to Run

1.  Ensure your **LM Studio server is running** with the **`Hunyuan-MT-7B-GGUF`** model loaded.
2.  Make sure your `translator` conda environment is active in your terminal.
3.  Run the application from your terminal:
    ```bash
    python translator.py
    ```
4.  Open your web browser and navigate to **`http://127.0.0.1:7860`**.
5.  Select your source and target languages, and start translating!

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any feedback or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
