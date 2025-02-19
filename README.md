# Sequential Thinking Paper

This repository contains the implementation of a Sequential Thinking Server for AI-powered article illustration. The server analyzes article content using a step-by-step approach to extract key information and generate appropriate visual suggestions.

## Features

- Content Analysis: Breaks down articles into meaningful segments
- Keyword Extraction: Identifies key themes and topics
- Emotion Analysis: Determines the emotional tone of content
- Visualization Suggestions: Recommends appropriate visualization types
- Image Prompt Generation: Creates prompts for AI image generation

## Architecture

The system is built using:
- FastAPI for the web server
- OpenAI's GPT-4 for content analysis
- Pydantic for data validation
- pytest for testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/sequential-thinking-paper.git
cd sequential-thinking-paper
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Usage

1. Start the server:
```bash
python run_servers.py
```

2. Send a POST request to analyze content:
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your article text here", "max_segments": 1, "analysis_type": "article"}'
```

## Testing

Run tests with:
```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{sequential-thinking,
  title={Sequential Thinking: A Step-by-Step Approach to AI Content Analysis},
  author={Your Name},
  year={2025},
  journal={Preprint}
}
```
