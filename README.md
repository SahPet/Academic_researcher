# Academic Researcher

An AI-powered tool for comprehensive academic research, leveraging multiple LLMs to gather, analyze and synthesize scientific literature.

## Features

- Multi-model approach using O3-mini, Gemini 2.0 Flash with thinking, and DeepSeekR1
- Iterative search with adaptive query generation
- Citation validation and metadata enrichment
- Comprehensive reference management
- Quality review and content revision
- Support for academic paper formats and citations

![Untitled_diagram_resized](https://github.com/user-attachments/assets/52012163-4251-4993-8858-7f5841433053)


## Prerequisites

- Python 3.8+
- Required API keys (see .env.example)
- Chrome/Chromium for web scraping capabilities

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your API keys
4. Install Chrome/Chromium for Selenium support

## Usage

Run the main script:
```bash
python research_crew.py
```

Enter your research question when prompted. The tool will:
1. Generate search queries
2. Gather relevant academic references
3. Synthesize content with proper citations
4. Validate and enrich references
5. Produce a final academic text

## Output Structure

Results are saved in `research_outputs/` with session-specific folders containing:
- Content drafts and final version
- Search results and reference data
- Citation validations
- Quality review feedback

## Contributing

Contributions welcome! Please read the contributing guidelines first.

## License

MIT License - see LICENSE file for details
