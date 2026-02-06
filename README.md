# OCR Benchmarking System

A comprehensive benchmarking system to compare OCR performance across three leading APIs:
- **Azure Document Intelligence** - Microsoft's document analysis service
- **Gemini 2.5 Flash** - Google's multimodal AI with vision
- **Llama 3.2 90B Vision** - Meta's open-source vision model (via Together AI)

## Features

✨ **Comprehensive Metrics**
- ⏱️ Execution time measurement
- 💰 Cost estimation per document
- 🎯 Accuracy scoring (with ground truth comparison)
- 📊 Token usage tracking

📈 **Rich Reporting**
- CSV exports with detailed results
- Summary statistics by API
- JSON output with full extracted data
- HTML reports with visualizations
- Performance comparison charts

🔄 **Flexible Schema Support**
- VAT Invoice extraction
- Medical claim form extraction
- Custom schema support

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-azure-key
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Google Gemini
GOOGLE_GEMINI_API_KEY=your-gemini-key

# Together AI (for Llama)
TOGETHER_API_KEY=your-together-key
```

### 3. Getting API Keys

#### Azure Document Intelligence
1. Go to [Azure Portal](https://portal.azure.com)
2. Create a "Document Intelligence" resource
3. Copy the key and endpoint from the resource

#### Google Gemini
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Copy the key

#### Together AI (Llama)
1. Sign up at [Together AI](https://www.together.ai/)
2. Go to API Keys section
3. Create and copy a new API key

## Usage

### Quick Start

Run the benchmark with default settings:

```bash
python run_benchmark.py
```

### Advanced Usage

Use the main benchmark class directly:

```python
from ocr_benchmark import OCRBenchmark

# Initialize with API keys
benchmark = OCRBenchmark(
    azure_key="your-key",
    azure_endpoint="your-endpoint",
    gemini_key="your-key",
    llama_key="your-key"
)

# Define input files and schemas
input_files = [
    "path/to/invoice.pdf",
    "path/to/claim_form.pdf"
]

schemas = {
    'invoice': "path/to/invoice_schema.json",
    'claim': "path/to/claim_schema.json"
}

# Optional: Add ground truth for accuracy measurement
ground_truth = {
    "invoice.pdf": {
        "seller": {"name": "Company ABC", ...},
        "buyer": {...},
        "invoice": {...}
    }
}

# Run benchmark
benchmark.run_benchmark(input_files, schemas, ground_truth)

# Generate reports
reports = benchmark.generate_report()
```

### Custom Schema

The system works with any JSON schema. Example:

```python
custom_schema = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "total_amount": {"type": "number"},
        "date": {"type": "string"}
    },
    "required": ["company_name", "total_amount"]
}

# Use in benchmark
benchmark.run_benchmark(
    files=["document.pdf"],
    schemas={"custom": custom_schema}
)
```

## Output Files

The benchmark generates several output files in `/mnt/user-data/outputs/`:

1. **`benchmark_results_[timestamp].csv`**
   - Detailed results for each API and file
   - Columns: api_name, file_name, execution_time, cost, accuracy, etc.

2. **`benchmark_summary_[timestamp].csv`**
   - Aggregated statistics by API
   - Mean, std, min, max for each metric

3. **`benchmark_detailed_[timestamp].json`**
   - Complete results including extracted data
   - Useful for detailed analysis and debugging

4. **`benchmark_report_[timestamp].html`**
   - Interactive HTML report with visualizations
   - Performance charts and comparison tables

5. **`benchmark_charts_[timestamp].png`**
   - Static charts comparing all APIs
   - Execution time, cost, accuracy, success rate

## Understanding Results

### Execution Time
- Lower is better
- Includes API call time + processing
- Azure typically fastest for simple extraction
- Gemini balances speed and accuracy

### Cost
- Based on current API pricing (as of Feb 2025)
- Azure: ~$1 per 1000 pages
- Gemini: ~$0.15 per 1M input tokens
- Llama: ~$0.15 per 1M tokens (via Together AI)

### Accuracy
- Measured by comparing to ground truth
- Field-by-field comparison
- Fuzzy matching for strings
- 0-100% scale

### Success Rate
- Percentage of successful extractions
- Failed extractions may indicate:
  - API errors
  - Schema validation failures
  - File format issues

## Troubleshooting

### Common Issues

**"Missing API key" error**
- Ensure `.env` file exists and has correct keys
- Check that keys are not expired

**"File not found" error**
- Verify file paths are correct
- Ensure files are uploaded to `/mnt/user-data/uploads/`

**Low accuracy scores**
- Add or improve ground truth data
- Check that schema matches document structure
- Review extracted_data in JSON output

**Azure timeout errors**
- Large PDF files may take longer
- Increase timeout in Azure client configuration

**Gemini rate limits**
- Free tier has limits (~15 requests/min)
- Add delays between requests if needed

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

benchmark = OCRBenchmark(...)
benchmark.run_benchmark(...)
```

## Performance Tips

1. **For best accuracy**: Use ground truth data for validation
2. **For speed**: Azure is typically fastest for simple extraction
3. **For cost**: Llama via Together AI often most economical
4. **For complex documents**: Gemini 2.5 Flash performs well

## Customization

### Adding New APIs

Extend the `OCRBenchmark` class:

```python
def benchmark_custom_api(self, file_path, schema, schema_type):
    start_time = time.time()
    result = BenchmarkResult(...)
    
    try:
        # Your API call here
        extracted_data = your_api_call(file_path, schema)
        
        result.success = True
        result.extracted_data = extracted_data
        result.execution_time = time.time() - start_time
        
    except Exception as e:
        result.error_message = str(e)
    
    return result
```

### Custom Accuracy Metrics

Override the `calculate_accuracy` method:

```python
class CustomBenchmark(OCRBenchmark):
    def calculate_accuracy(self, extracted, ground_truth):
        # Your custom scoring logic
        return score
```

## Project Structure

```
.
├── ocr_benchmark.py          # Main benchmarking class
├── run_benchmark.py          # Simple runner script
├── llama_openai_api.py      # Alternative Llama implementation
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
└── README.md                # This file
```

## API Pricing Reference (Feb 2025)

| API | Model | Input Cost | Output Cost | Notes |
|-----|-------|-----------|-------------|-------|
| Azure | Document Intelligence | $1/1K pages | - | Flat rate per page |
| Gemini | 2.5 Flash | $0.15/1M tokens | $0.60/1M tokens | Vision-enabled |
| Llama | 3.2 90B Vision | $0.15/1M tokens | $0.15/1M tokens | Via Together AI |

*Prices subject to change - check provider websites for current rates*

## Requirements

- Python 3.8+
- Internet connection for API calls
- API keys from Azure, Google, and Together AI
- PDF files for testing

## License

This benchmarking tool is provided as-is for evaluation purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API provider documentation
3. Verify API keys and quotas
4. Check output logs for detailed error messages

## Contributing

To add support for new APIs or improve accuracy measurement:
1. Extend the `OCRBenchmark` class
2. Add corresponding pricing information
3. Update the README with new API details

## Changelog

### Version 1.0.0
- Initial release
- Support for Azure, Gemini, and Llama
- Comprehensive metrics and reporting
- HTML visualization
- Schema validation
- Accuracy measurement
