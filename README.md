# StyleShift - AI-Powered Writing Style Transfer

A sophisticated style transfer system that analyzes writing styles using 100+ linguistic features and transfers them to new text using LLM-powered transformation.

## Features

- **Comprehensive Style Analysis**: Analyzes 100+ linguistic features across 7 categories:
  - Lexical features (word length, vocabulary richness, rare words)
  - Syntactic patterns (sentence structure, voice, tense)
  - Sentiment & emotion (polarity, subjectivity, VADER scores)
  - Readability metrics (Flesch-Kincaid, Gunning Fog, SMOG)
  - Part-of-speech distributions
  - Punctuation patterns
  - Discourse markers and coherence

- **Web Interface**: User-friendly Flask web application
- **LLM Integration**: Uses Ollama for style transfer with local models

## Prerequisites

- Python 3.8 or higher
- Ollama (for LLM-based style transfer)
- 4GB+ RAM recommended
- Windows, macOS, or Linux

## Installation

### Step 1: Clone the Repository

```powershell
git clone https://github.com/DarkMatterCompiler/StyleShift.git
cd StyleShift
```

### Step 2: Install Python Dependencies

```powershell
pip install flask spacy textstat textblob vaderSentiment transformers torch numpy matplotlib requests
```

### Step 3: Download spaCy Language Model

```powershell
python -m spacy download en_core_web_sm
```

### Step 4: Install and Set Up Ollama

#### Windows

1. **Download Ollama**:
   - Visit [https://ollama.com/download](https://ollama.com/download)
   - Download the Windows installer
   - Run the installer and follow the setup wizard

2. **Start Ollama**:
   - Ollama runs as a service after installation
   - Verify it's running by opening a new PowerShell window:
   ```powershell
   ollama --version
   ```

3. **Pull the Required Model**:
   ```powershell
   ollama pull gemma3:4b
   ```
   
   **Note**: The default model is `gemma3:4b`. If you want to use a different model (like `llama2`, `mistral`, etc.), you'll need to modify `model.py` line 817:
   ```python
   def __init__(self, ollama_model: str = "your-model-name"):
   ```

4. **Verify Ollama is Running**:
   ```powershell
   curl http://localhost:11434
   ```
   You should see: `Ollama is running`

#### macOS

```bash
# Install using Homebrew
brew install ollama

# Start Ollama service
ollama serve &

# Pull the model
ollama pull gemma3:4b
```

#### Linux

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve &

# Pull the model
ollama pull gemma3:4b
```

### Step 5: Verify Installation

Test that all dependencies are installed:

```powershell
python -c "import flask, spacy, textstat, textblob, vaderSentiment, transformers; print('All dependencies installed successfully!')"
```

## Running the Application

### Method 1: Run the Web Application (Recommended)

1. **Start the Flask server**:
   ```powershell
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:5002
   ```

3. **Using the Web Interface**:
   - **Step 1**: Paste stylized text (example from a specific author) in the "Stylized Text" field
   - **Step 2**: Give your style a name (e.g., "Shakespeare", "Hemingway")
   - **Step 3**: Click "Analyze Style" to extract the style vector
   - **Step 4**: Enter your original text in the "Original Text" field
   - **Step 5**: Click "Transfer Style" to apply the style

### Method 2: Run Command-Line Analysis

For analyzing a text corpus and generating style vectors:

1. **Prepare your training data**:
   - Create a folder: `data/`
   - Add a file: `data/style_train.txt`
   - Fill it with sample text from your target author (at least 500-1000 words recommended)

2. **Run the analysis**:
   ```powershell
   python model.py
   ```

3. **View results**:
   - Style vector saved to: `style_vectors/edgar_allan_poe.json`
   - Visualization saved as: `style_analysis.png`
   - Console output shows comprehensive style metrics

## Configuration Options

### Change Host/Port (Web App)

By default, the app runs on `127.0.0.1:5002`. To customize:

```powershell
# Run on a different port
$env:FLASK_RUN_PORT='8080'; python app.py

# Allow network access (accessible from other machines)
$env:FLASK_RUN_HOST='0.0.0.0'; $env:FLASK_RUN_PORT='5002'; python app.py

# Disable debug mode
$env:FLASK_DEBUG='False'; python app.py
```

### Change Ollama Model

Edit `model.py` line 817 to use a different model:

```python
def __init__(self, ollama_model: str = "llama2"):  # or "mistral", "codellama", etc.
```

Available models:
- `gemma3:4b` (default, good balance of performance and speed)
- `llama2` (more powerful, requires more RAM)
- `mistral`
- `phi`
- See full list: [https://ollama.com/library](https://ollama.com/library)

## Example Workflow

### Example 1: Analyze Edgar Allan Poe's Style

1. Create `data/style_train.txt` with Poe's text:
```text
Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.
```

2. Run analysis:
```powershell
python model.py
```

3. Check output:
   - `style_vectors/edgar_allan_poe.json` - Complete style profile
   - `style_analysis.png` - Visual representation
   - Console shows 100+ features

### Example 2: Web-Based Style Transfer

1. Start the app:
```powershell
python app.py
```

2. In browser (`http://127.0.0.1:5002`):
   - **Stylized Text**: Paste several paragraphs of your target style
   - **Style Name**: "Poe"
   - Click **Analyze Style**
   - **Original Text**: "I went to the store. It was sunny."
   - Click **Transfer Style**
   - See transformed text in Poe's Gothic style!

## Troubleshooting

### Issue: "Cannot connect to Ollama"

**Solution**:
```powershell
# Check if Ollama is running
ollama --version

# If not running, start it (Windows)
# Ollama runs as a service, restart the Ollama application

# On macOS/Linux
ollama serve &
```

### Issue: "Model not found"

**Solution**:
```powershell
# Pull the required model
ollama pull gemma3:4b
```

### Issue: "spaCy model not found"

**Solution**:
```powershell
python -m spacy download en_core_web_sm
```

### Issue: Port 5002 already in use

**Solution**:
```powershell
# Use a different port
$env:FLASK_RUN_PORT='8080'; python app.py
```

### Issue: "memory layout cannot be allocated" (Ollama Memory Error)

This is the most common error with `gemma3:4b` and larger models. The model is trying to load more data than your system can handle.

**Solution 1 - Use a Smaller Model** (Recommended):
```powershell
# Pull and use a lighter model
ollama pull gemma2:2b

# Then edit model.py line 817 to change the default model:
# def __init__(self, ollama_model: str = "gemma2:2b"):
```

**Solution 2 - Reduce Input Text Length**:
- Use shorter text samples (under 100 words for analysis)
- Break longer texts into smaller chunks
- The app now automatically truncates long prompts

**Solution 3 - Free Up System Memory**:
```powershell
# Stop Ollama completely
ollama stop

# Close other applications (browsers, etc.)

# Restart Ollama
# On Windows: Restart the Ollama application
# On macOS/Linux: ollama serve &

# Try again with fresh memory
```

**Solution 4 - Increase Ollama Memory Limit** (Advanced):
```powershell
# Set environment variable before starting Ollama
# Windows PowerShell:
$env:OLLAMA_MAX_LOADED_MODELS=1
$env:OLLAMA_NUM_PARALLEL=1

# Then restart Ollama application
```

### Issue: Out of memory (General)

**Solution**:
- Use a smaller Ollama model (e.g., `gemma2:2b` instead of `gemma3:4b` or `llama2`)
- Reduce training corpus size
- Close other applications
- Restart your computer to free up memory

## Project Structure

```
StyleShift/
├── model.py              # Core style analysis & transfer engine
├── app.py               # Flask web application
├── templates/
│   └── index.html       # Web interface
├── static/
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
├── data/
│   └── style_train.txt  # Training corpus (user-provided)
├── style_vectors/       # Generated style profiles
└── README.md            # This file
```

## Technical Details

### Style Vector Features (100+)

- **Lexical**: Word length, vocabulary richness, hapax legomena, intensifiers
- **Syntactic**: Sentence complexity, parse tree depth, voice, tense
- **Sentiment**: Polarity, subjectivity, VADER scores, emotions
- **Readability**: Flesch-Kincaid, Gunning Fog, Coleman-Liau, SMOG
- **POS**: Adjective/adverb/noun/verb ratios, pronouns, determiners
- **Punctuation**: Comma, semicolon, dash, ellipsis frequencies
- **Discourse**: Markers, rhetorical questions, narrative vs analytical

### How It Works

1. **Analysis Phase**:
   - Input: Corpus of text in target style
   - Process: Extract 100+ linguistic features using spaCy, TextBlob, VADER
   - Output: StyleVector JSON with comprehensive profile

2. **Transfer Phase**:
   - Input: Original text + Style vector
   - Process: Create detailed style prompt → Send to Ollama LLM
   - Output: Text rewritten in target style

## Performance Tips

- **Faster analysis**: Use smaller corpus (500-1000 words minimum)
- **Better results**: Use larger, representative corpus (5000+ words)
- **Lower memory**: Use `gemma2:2b` model (lighter than `gemma3:4b`)
- **Higher quality**: Use `gemma3:4b` (default), `llama2`, or `mistral` (requires more RAM)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit pull requests or open issues.

## Support

For issues or questions:
- GitHub Issues: [https://github.com/DarkMatterCompiler/StyleShift/issues](https://github.com/DarkMatterCompiler/StyleShift/issues)
- Email: Contact repository owner

## Acknowledgments

- spaCy for NLP processing
- Ollama for local LLM inference
- VADER for sentiment analysis
- TextBlob for linguistic features

---

**Happy Style Transferring! 🎨✍️**
