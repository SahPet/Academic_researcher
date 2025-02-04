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
```bash
git clone https://github.com/SahPet/Academic_researcher.git
cd Academic_researcher
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Change the name  `.env.example` to `.env` and add your API keys (**required** unless you comment out parts of the code):
- `OPENAI_API_KEY`: Obtain from [OpenAI API Keys](https://platform.openai.com/account/api-keys)
- `GEMINI_API_KEY`: Obtain from [Google Gemini API Keys](https://ai.google.dev/gemini-api/docs/get-started)
- `DEEPSEEK_API_KEY`: Obtain from [DeepSeek API Keys](https://deepseek.io/api-keys)
- `SCRAPINGBEE_API_KEY`: Obtain from [ScrapingBee API Keys](https://www.scrapingbee.com/get-started/)
- `PARSEHUB_API_KEY` and `PARSEHUB_PROJECT_TOKEN`: Obtain from [ParseHub API Documentation](https://parsehub.com/docs#api)

4. Install Chrome/Chromium for Selenium support

## Usage


**Optional parameters (change in the main py file):**
- BASE_OUTPUT_FOLDER: Default is C:\research_outputs
- MAX_SEARCH_ROUNDS: Default is 7
- MAX_REFERENCES: Default is 12

Running the main script:
```bash
python research_crew_NO_CREWAI_simplified_github.py
```

When prompted, enter your research question. The tool will:
- Generate search queries
- Gather relevant academic references
- Synthesize content with proper citations
- Validate (fulltext download) and add extra references if neeed
- Produce and iteratively improve the academic text

## Output Structure

Results are saved in `C:/research_outputs/` with subfolders containing:
- Content drafts and final version
- Search results and reference data
- Citation validations
- Quality review feedback

## Contributing

Contributions welcome!

## License

MIT License - see LICENSE file for details
