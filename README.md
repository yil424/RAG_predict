Online demo without LLM: https://ragpredict-nvgcqifzvpvudff4kvdkuh.streamlit.app

Full offline demo with local LLM: see ‘Local demo with LLM’ below.

## Local demo with LLM (Ollama + Llama 3.1 8B)

To fully reproduce the RAG + early-warning web UI **with local LLM answers**, please follow these steps on a Windows 10/11 machine:

1. **Install Python (3.10–3.13)**  
   - Download from https://www.python.org/downloads/  
   - During installation, tick “Add Python to PATH”.

2. **Install Ollama**  
   - Download the Windows installer from https://ollama.com/download  
   - After installation, open **Command Prompt** and run:
     ```bash
     ollama pull llama3.1:8b
     ```
     This downloads the Llama 3.1 8B model used in our app.

3. **Download this repository**
   - Either use:
     ```bash
     git clone https://github.com/yil424/RAG_predict.git
     ```
   - Or click “Code → Download ZIP” on GitHub and unzip it.

4. **Run the demo by double-clicking `run_app.bat`**
   - Inside the project folder (`RAG_predict`), double-click `run_app.bat`.
   - The script will:
     - create a Python virtual environment (if not existing),
     - install all required packages,
     - launch the Streamlit app.
   - A browser window should automatically open at:
     `http://localhost:8501`

5. **Using the app**
   - The RAG + risk analytics UI will appear.
   - The chat panel will now use the local **Llama 3.1 8B** model via Ollama, completely offline.
