# CSV AI Analytics

A powerful platform for analyzing, visualizing, and extracting insights from CSV data with integrated AI capabilities.

![CSV AI Analytics Architecture](architecture-diagram.png)

## Overview

CSV AI Analytics is a comprehensive data analysis platform designed to help users explore, process, visualize, and gain insights from CSV data. The application integrates rule-based and AI-powered assistants to provide intelligent recommendations and automate various aspects of the data analysis process.

## Architecture

The application is built with a modular architecture organized around functional units:

### Core Functional Units

- **Data Management**: Manages data loading, storage, and state management
- **File Handling**: Processes file uploads, downloads, and format conversions
- **UI Components**: Reusable UI elements shared across the application
- **Utilities**: Helper functions for common tasks

### Data Processing Units

- **Data Overview**: Provides summary statistics and initial data insights
- **Data Processor**: Handles data cleaning, transformation, and preparation
- **Data Analyzer**: Runs statistical analysis and data profiling
- **Data Exporter**: Exports processed data in various formats

### Analysis & Visualization

- **Visualization**: Creates charts, graphs, and visual data representations
- **Analysis**: Performs in-depth statistical analysis and modeling
- **Reports**: Generates formatted reports from analysis results
- **Dashboard**: Creates interactive dashboards for data monitoring

### AI Assistant Units

- **GenAI Assistant**: Leverages Claude API for natural language insights
- **AI Assistant**: Rule-based system for automated data recommendations

### Project Management

- **Project Manager**: Saves and loads analysis projects and manages workflows

### Data Storage (Session State)

- **Original Data**: The uploaded/imported raw data
- **Processed Data**: Cleaned and transformed datasets
- **Projects**: Saved analysis projects and configurations
- **Chat History**: Record of interactions with AI assistants

### External APIs

- **Claude**: Integration with Anthropic's Claude API for advanced NLP capabilities

## Key Features

- Intuitive user interface with organized tabs for different analysis tasks
- Comprehensive data processing and cleaning capabilities
- Advanced visualization tools for data exploration
- Integrated AI assistance for data insights and recommendations
- Project management for saving and retrieving analysis sessions
- Export capabilities for sharing results in various formats

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Pandas, NumPy, and other required data analysis libraries
- API keys for Claude (if using GenAI features)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/csv-ai-analytics.git
   cd csv-ai-analytics
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Configure your API keys (create a `.env` file based on `.env.example`)

### Running the Application

Start the application with:

```
streamlit run app.py
```

## Usage

1. **Data Input**: Upload your CSV file or use sample datasets
2. **Data Overview**: Get initial insights and summary statistics
3. **Data Processing**: Clean and transform your data as needed
4. **Visualization**: Create visualizations to explore your data
5. **Analysis**: Run in-depth analysis on your processed data
6. **AI Insights**: Ask the AI assistants for recommendations
7. **Reports**: Generate and export reports of your findings
8. **Projects**: Save your work for future sessions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the powerful web application framework
- Anthropic for the Claude API integration
- The open-source data science community for various tools and libraries