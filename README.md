![stylized_brain_with_glowing_circuits_half_size](https://github.com/user-attachments/assets/52ea808a-c7ce-4ed9-a109-ecdbd972f52d)

# Academic Researcher

An AI-powered tool for writing well cited (and citation checked) academic paragraphs. It leverages multiple LLMs to gather, analyze and synthesize scientific literature.

## Features

- Multi-model approach using:
  - [O3-mini](https://openai.com/index/openai-o3-mini/) - efficient reasoning LLM
  - [Google Gemini 2.0 Flash with thinking](https://ai.google.dev/gemini) - Latest Gemini model with "thinking".
  - [DeepSeekR1](https://docs.fireworks.ai/api-reference/introduction) DeepSeek's reasoning LLM via the Fireworks.ai API (stable, fast and secure compared to the original DeepSeek API).
- Iterative search with adaptive query generation
- Fulltext citation by citation validation and metadata enrichment
- Comprehensive reference management
- Quality review and content revision

![Untitled_diagram_resized](https://github.com/user-attachments/assets/52012163-4251-4993-8858-7f5841433053)

## Prerequisites

- Python 3.8+
- Required API keys (see .env.example)
- Chrome/Chromium for web scraping capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SahPet/Academic_researcher.git
cd Academic_researcher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Change the name `.env.example` to `.env` and add your API keys (**required** unless you comment out parts of the code):
- `OPENAI_API_KEY`: Obtain from [OpenAI API Keys](https://platform.openai.com/account/api-keys)
- `GEMINI_API_KEY`: Obtain from [Google Gemini API Keys](https://ai.google.dev/gemini-api/docs/get-started)
- `FIREWORKS_API_KEY`: Fast and stable DeepSeekR1 API - obtain from [Fireworks API Keys](https://docs.fireworks.ai/api-reference/introduction)
- `SCRAPINGBEE_API_KEY`: Obtain from [ScrapingBee API Keys](https://www.scrapingbee.com/get-started/)
- `PARSEHUB_API_KEY` and `PARSEHUB_PROJECT_TOKEN`: Obtain from [ParseHub API Documentation](https://parsehub.com/docs#api)

4. Install Chrome/Chromium for Selenium support

## Usage

**Optional parameters** (change in the main py file):
- `BASE_OUTPUT_FOLDER`: Default is `C:\research_outputs`
- `MAX_SEARCH_ROUNDS`: Default is 7
- `MAX_REFERENCES`: Default is 12

Running the main script:
```bash
python research_crew_NO_CREWAI_simplified_github.py
```

When prompted, enter your research question. The tool will:
- Generate search queries
- Gather relevant academic references
- Synthesize content with proper citations
- Validate (fulltext download) and add extra references if needed
- Produce and iteratively improve the academic text

## Output Structure

### Example Input
```
What are the current best strategies for auto annotation of semantic segmentation?
```

### Example Output
<details>
<summary>Click to expand example output</summary>

Current best strategies for auto annotation of semantic segmentation datasets center on minimizing the labor-intensive process of dense pixel-level labeling by leveraging a combination of weak and noisy supervision, active learning, and human-in-the-loop correction. For instance, recent frameworks incorporating semi-supervised learning with uncertainty-aware active sampling have demonstrated that segmentation models can achieve competitive Dice scores while significantly reducing the manual annotation workload, with some studies suggesting reductions on the order of 50‒70% compared to fully supervised methods (Wang et al., 2021; Zhang et al., 2025).

Additionally, empirical research indicates that using point-based and image-level annotations can accelerate the labeling process and help mitigate common human errors without compromising segmentation accuracy (Fernández-Moreno, 2023; Zhang et al., 2025). Emerging zero-shot approaches also harness self-supervised techniques—as evidenced by the scalability of masked autoencoders for feature learning—to automatically generate annotations without extensive manual input (He et al., 2022; Xie et al., 2022).

Furthermore, the integration of automated pre-annotation models, such as the Segment Anything Model (SAM), with selective human verification has shown promise for efficiently handling uncertain or complex cases, although further research is needed to validate its generalizability across diverse imaging modalities (Kirillov et al., 2023).

#### References:
- Fernández-Moreno, M. (2023). Exploring the trade-off between performance and annotation in deep learning: An engineering perspective. Engineering Applications of Artificial Intelligence. Retrieved from https://openreview.net/pdf?id=jMiZegbLUe
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 15979‒15988). https://doi.org/10.1109/CVPR52688.2022.01553
- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. (2023). Segment Anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4015‒4026).
- Wang, S., Li, C., Liu, Z., & Wang, R. (2021). Annotation-efficient deep learning for automatic medical image segmentation. Nature Communications, 12(1). Retrieved from https://www.nature.com/articles/s41467-021-26216-9
- Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., & Hu, H. (2022). Simmim: A simple framework for masked image modeling. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 9643‒9653). https://doi.org/10.1109/CVPR52688.2022.00943
- Zhang, Y., Zhao, S., Gu, H., & Mazurowski, M. A. (2025). How to efficiently annotate images for best-performing deep learning-based segmentation models: An empirical study with weak and noisy annotations and Segment Anything Model. Journal of Imaging Informatics in Medicine. Retrieved from https://pubmed.ncbi.nlm.nih.gov/39843720

</details>

Results are saved in `C:/research_outputs/` with subfolders containing:
- Content drafts and final version
- Search results and reference data
- Citation validations
- Quality review feedback

## Contributing

Contributions welcome!

## License

MIT License - see LICENSE file for details
