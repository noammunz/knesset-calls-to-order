# Knesset Calls to Order Classifier

This project leverages machine learning and natural language processing (NLP) techniques to automatically identify and classify "calls to order" within Israeli Parliament (Knesset) session transcripts. Our system analyzes parliamentary discourse dynamics by detecting when speakers are called to maintain decorum during debates.

## Project Overview

The Knesset Calls to Order Classifier systematically processes session transcripts using advanced NLP algorithms to:
- Detect instances where the chairperson calls for order
- Classify these interventions based on context, urgency, and other parameters
- Provide insights into parliamentary discourse patterns and procedural enforcement

## Key Features

- **Automated Transcript Processing**: Convert raw Knesset protocols into structured, analyzable text
- **ML-Based Classification**: Implement supervised learning models to identify calls to order
- **Context Analysis**: Examine the relationship between conversation length and procedural interventions
- **Multi-dimensional Feature Analysis**: Incorporate speaker, chairperson, and committee vectors for enhanced prediction accuracy

## Data Sources and Processing

### Source Data
The project uses official Knesset meeting protocols available at:
[Knesset Meeting Protocols](https://production.oknesset.org/pipelines/data/committees/meeting_protocols_text/files/)

### Data Processing Pipeline
Our preprocessing workflow transforms raw protocols into structured data suitable for ML analysis:

![Preprocessing Steps](https://github.com/nogaschw/Call-to-order/assets/80199057/0a9bf47f-4816-4203-8d3d-aea7769e4882)

The complete preprocessing implementation can be found in:
- `Create_protocol_txt.py`: Initial text extraction and formatting
- `Parser.py`: Structural parsing and feature extraction

## Analysis and Results

### Contextual Window Analysis
We investigated correlations between conversation length and the frequency of calls to order, revealing patterns in parliamentary discourse management.

### Feature Enhancement
Our models incorporate:
- Speaker identity vectors
- Chairperson characteristics
- Committee-specific contextual information
- Temporal patterns within sessions

### Results and Findings
Detailed experimental results and model performance metrics are documented in `results.ipynb`, demonstrating the effectiveness of our classification approach.

### Statistical Analysis
Comprehensive statistical insights about the dataset and call-to-order patterns are available in `statistic.ipynb`, including:
- Frequency distributions
- Temporal trends
- Committee-specific patterns
- Speaker correlation analysis

## Getting Started

1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the preprocessing pipeline: `python Create_protocol_txt.py` followed by `python Parser.py`
4. Explore the Jupyter notebooks for analysis and results

## Future Work

- Implement real-time classification for live Knesset sessions
- Expand the model to classify different types of parliamentary interventions
- Develop a visualization dashboard for parliamentary discourse analysis
- Comparative analysis with other parliamentary systems

## Contributing

We welcome contributions to enhance this project. Please feel free to submit pull requests or open issues for discussion.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.
