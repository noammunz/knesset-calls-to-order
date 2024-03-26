# Knesset Calls to Order

This project aims to utilize machine learning and natural language processing (NLP) techniques to automatically classify instances of "calls to order" within the transcripts of Knesset (Israeli Parliament) sessions. By analyzing session transcripts, our goal is to identify various forms of order calls, facilitating a deeper understanding of parliamentary discourse dynamics.

## Project Overview

The Knesset Calls to Order Classifier is designed to parse through session transcripts, employing NLP algorithms to detect and classify instances where a speaker calls for order. This process involves semantic analysis, pattern recognition, and supervised learning models trained on a labeled dataset of transcript segments.

## Project Features

- **Transcript Parsing**: Automated processing of Knesset session transcripts to extract text for analysis.
- **Machine Learning Classification**: Utilization of NLP and machine learning models to classify segments as calls to order, with categorization based on the context and urgency.

## Data

**Original Transcripts**:
The original transcripts of Knesset session meetings are pivotal to our analysis. These documents provide the raw textual content from which we extract calls to order. The original meeting protocols can be accessed at the following location:

[Knesset Meeting Protocols](https://production.oknesset.org/pipelines/data/committees/meeting_protocols_text/files/)

**Preprocessing**:
Our preprocessing steps are essential for preparing the data for analysis. The preprocessing steps are defined in the following image and can be found in the files "Create_protocol_txt" and "Parser":
![Preprocessing Steps](https://github.com/nogaschw/Call-to-order/assets/80199057/0a9bf47f-4816-4203-8d3d-aea7769e4882)

## Our Work

**Windows Context**:
We investigated whether there is any correlation between long conversations and calls to order within Knesset sessions. This analysis provides insights into the dynamics of parliamentary discourse.

**Feature Enhancement**:
We augmented our analysis by incorporating vectors of the chairperson, speaker, and committee involved in each segment. This enriched feature set allows for a more nuanced understanding of the context surrounding calls to order.

**Experience Results**:
The results of our experiments and analyses are documented in "results.ipynb". These findings shed light on the effectiveness of our classification models and provide insights into the patterns observed in Knesset session transcripts.

**Basic Statistics**:
We conducted basic statistical analysis on the data to gain a better understanding of its characteristics. This analysis is documented in "statistic.ipynb" and provides valuable insights into the distribution and properties of the dataset.

Feel free to reach out if you have any questions or need further clarification on any aspect of our project!
