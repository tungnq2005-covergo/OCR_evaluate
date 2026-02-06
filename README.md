# OCR Benchmarking System

A comprehensive benchmarking system to compare OCR performance across three leading APIs:
- **Azure Document Intelligence** - Microsoft's document analysis service
- **Llama Cloud OCR** - LlamaIndex service


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


# Together AI (for Llama)
LLAMA_CLOUD_API_KEY=your-llama-cloud-key
```

### 3. Getting API Keys


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


