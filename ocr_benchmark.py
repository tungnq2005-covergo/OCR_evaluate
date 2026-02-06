"""
OCR Benchmarking System
Compares Azure Document Intelligence, Gemini 2.5, and Llama Vision APIs
Measures: Time, Cost, and Accuracy
"""

import json
import time
import base64
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# API clients
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import google.generativeai as genai
import anthropic


@dataclass
class BenchmarkResult:
    """Store results for a single OCR operation"""
    api_name: str
    file_name: str
    schema_type: str
    execution_time: float
    estimated_cost: float
    token_count: int
    success: bool
    error_message: str = ""
    extracted_data: Dict = None
    accuracy_score: float = 0.0


class OCRBenchmark:
    """Main benchmarking class"""
    
    # Pricing information (as of Feb 2025)
    PRICING = {
        'azure': {
            'per_page': 0.001,  # $1 per 1000 pages
        },
        'gemini': {
            'input_per_1k': 0.00015,  # Gemini 2.5 Flash pricing
            'output_per_1k': 0.0006,
        },
        'llama': {
            'per_page': 0.003,  # LlamaCloud pricing ~$0.003 per page
        }
    }
    
    def __init__(self, azure_key: str, azure_endpoint: str, gemini_key: str, llama_key: str):
        """Initialize API clients"""
        # Azure Document Intelligence
        self.azure_client = DocumentIntelligenceClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key)
        )
        
        # Gemini
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Llama (via Anthropic-style API or other provider)
        self.llama_key = llama_key
        
        self.results: List[BenchmarkResult] = []
    
    def load_pdf_as_base64(self, file_path: str) -> str:
        """Load PDF file and convert to base64"""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def load_schema(self, schema_path: str) -> Dict:
        """Load JSON schema"""
        with open(schema_path, 'r') as f:
            return json.load(f)
    
    def create_extraction_prompt(self, schema: Dict) -> str:
        """Create prompt for structured extraction"""
        prompt = f"""Extract information from this document according to the following JSON schema.
Return ONLY valid JSON matching this schema, no additional text.

Schema:
{json.dumps(schema, indent=2)}

Instructions:
- Extract all fields as specified in the schema
- Follow all validation rules and enums
- Return properly formatted JSON
- If a field is not found, use null or empty string as appropriate
"""
        return prompt
    
    # ============= AZURE DOCUMENT INTELLIGENCE =============
    
    def benchmark_azure(self, file_path: str, schema: Dict, schema_type: str) -> BenchmarkResult:
        """Benchmark Azure Document Intelligence"""
        start_time = time.time()
        result = BenchmarkResult(
            api_name="Azure Document Intelligence",
            file_name=Path(file_path).name,
            schema_type=schema_type,
            execution_time=0,
            estimated_cost=0,
            token_count=0,
            success=False
        )
        
        try:
            with open(file_path, 'rb') as f:
                poller = self.azure_client.begin_analyze_document(
                    "prebuilt-layout",
                    analyze_request=f,
                    content_type="application/pdf"
                )
                azure_result = poller.result()
            
            # Extract text content
            full_text = ""
            for page in azure_result.pages:
                for line in page.lines:
                    full_text += line.content + "\n"
            
            # Use Gemini to structure the extracted text according to schema
            prompt = f"""{self.create_extraction_prompt(schema)}

Extracted Text:
{full_text}
"""
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif response_text.startswith('```'):
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            extracted_data = json.loads(response_text)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            page_count = len(azure_result.pages)
            estimated_cost = page_count * self.PRICING['azure']['per_page']
            
            # Add Gemini cost for structuring
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                estimated_cost += (
                    (input_tokens / 1000) * self.PRICING['gemini']['input_per_1k'] +
                    (output_tokens / 1000) * self.PRICING['gemini']['output_per_1k']
                )
                result.token_count = input_tokens + output_tokens
            
            result.execution_time = execution_time
            result.estimated_cost = estimated_cost
            result.success = True
            result.extracted_data = extracted_data
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.error_message = str(e)
        
        return result
    
    # ============= GEMINI 2.5 =============
    
    def benchmark_gemini(self, file_path: str, schema: Dict, schema_type: str) -> BenchmarkResult:
        """Benchmark Gemini 2.5 with vision"""
        start_time = time.time()
        result = BenchmarkResult(
            api_name="Gemini 2.5 Flash",
            file_name=Path(file_path).name,
            schema_type=schema_type,
            execution_time=0,
            estimated_cost=0,
            token_count=0,
            success=False
        )
        
        try:
            # Upload file to Gemini
            uploaded_file = genai.upload_file(file_path)
            
            # Wait for processing
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                raise Exception("File processing failed")
            
            # Generate content with schema
            prompt = self.create_extraction_prompt(schema)
            
            response = self.gemini_model.generate_content(
                [uploaded_file, prompt],
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            
            # Parse response
            extracted_data = json.loads(response.text)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                estimated_cost = (
                    (input_tokens / 1000) * self.PRICING['gemini']['input_per_1k'] +
                    (output_tokens / 1000) * self.PRICING['gemini']['output_per_1k']
                )
                result.token_count = input_tokens + output_tokens
            
            result.execution_time = execution_time
            result.estimated_cost = estimated_cost
            result.success = True
            result.extracted_data = extracted_data
            
            # Cleanup
            genai.delete_file(uploaded_file.name)
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.error_message = str(e)
        
        return result
    
    # ============= LLAMA CLOUD =============
    
    def benchmark_llama(self, file_path: str, schema: Dict, schema_type: str) -> BenchmarkResult:
        """Benchmark Llama via LlamaCloud (LlamaParse)"""
        start_time = time.time()
        result = BenchmarkResult(
            api_name="Llama (LlamaCloud)",
            file_name=Path(file_path).name,
            schema_type=schema_type,
            execution_time=0,
            estimated_cost=0,
            token_count=0,
            success=False
        )
        
        try:
            from llama_parse import LlamaParse
            
            # Initialize parser
            parser = LlamaParse(
                api_key=self.llama_key,
                result_type="markdown",
                verbose=False
            )
            
            # Parse document
            documents = parser.load_data(file_path)
            
            if not documents or len(documents) == 0:
                raise Exception("No content extracted from document")
            
            # Combine all pages
            full_text = "\n\n".join([doc.text for doc in documents])
            
            # Use Gemini to structure the extracted text according to schema
            prompt = f"""{self.create_extraction_prompt(schema)}

Extracted Text:
{full_text}
"""
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            
            # Parse JSON response
            extracted_data = json.loads(response.text)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            
            # LlamaCloud pricing (approximate - $0.003 per page)
            page_count = len(documents)
            estimated_cost = page_count * 0.003
            
            # Add Gemini structuring cost
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                estimated_cost += (
                    (input_tokens / 1000) * self.PRICING['gemini']['input_per_1k'] +
                    (output_tokens / 1000) * self.PRICING['gemini']['output_per_1k']
                )
                result.token_count = input_tokens + output_tokens
            
            result.execution_time = execution_time
            result.estimated_cost = estimated_cost
            result.success = True
            result.extracted_data = extracted_data
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.error_message = str(e)
        
        return result
    
    # ============= ACCURACY MEASUREMENT =============
    
    def calculate_accuracy(self, extracted: Dict, ground_truth: Dict) -> float:
        """Calculate accuracy score by comparing extracted data with ground truth"""
        if not extracted or not ground_truth:
            return 0.0
        
        def compare_values(v1, v2) -> float:
            """Compare two values and return similarity score"""
            if v1 == v2:
                return 1.0
            if v1 is None or v2 is None:
                return 0.0
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # For numbers, check if they're close
                if abs(v1 - v2) / max(abs(v1), abs(v2), 1) < 0.01:
                    return 1.0
                return 0.0
            if isinstance(v1, str) and isinstance(v2, str):
                # For strings, use fuzzy matching
                from difflib import SequenceMatcher
                return SequenceMatcher(None, v1.lower(), v2.lower()).ratio()
            return 0.0
        
        def compare_dicts(d1, d2, weight=1.0) -> Tuple[float, float]:
            """Recursively compare dictionaries"""
            total_score = 0.0
            total_weight = 0.0
            
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                v1 = d1.get(key)
                v2 = d2.get(key)
                
                if isinstance(v1, dict) and isinstance(v2, dict):
                    score, w = compare_dicts(v1, v2, weight)
                    total_score += score
                    total_weight += w
                elif isinstance(v1, list) and isinstance(v2, list):
                    # Compare lists element by element
                    max_len = max(len(v1), len(v2))
                    if max_len > 0:
                        list_score = sum(
                            compare_values(v1[i] if i < len(v1) else None,
                                         v2[i] if i < len(v2) else None)
                            for i in range(max_len)
                        ) / max_len
                        total_score += list_score * weight
                        total_weight += weight
                else:
                    total_score += compare_values(v1, v2) * weight
                    total_weight += weight
            
            return total_score, total_weight
        
        score, weight = compare_dicts(extracted, ground_truth)
        return (score / weight * 100) if weight > 0 else 0.0
    
    # ============= MAIN BENCHMARK RUNNER =============
    
    def run_benchmark(
        self,
        input_files: List[str],
        schemas: Dict[str, str],
        ground_truth: Dict[str, Dict] = None
    ):
        """Run complete benchmark across all APIs and files"""
        print("=" * 80)
        print("OCR BENCHMARK STARTED")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        for file_path in input_files:
            file_name = Path(file_path).name
            
            # Determine which schema to use
            if 'claim' in file_name.lower():
                schema_path = schemas.get('claim')
                schema_type = 'claim'
            else:
                schema_path = schemas.get('invoice')
                schema_type = 'invoice'
            
            schema = self.load_schema(schema_path)
            
            print(f"\n📄 Processing: {file_name} (Schema: {schema_type})")
            print("-" * 80)
            
            # Benchmark each API
            apis = [
                ('Azure', self.benchmark_azure),
                ('Gemini', self.benchmark_gemini),
                ('Llama', self.benchmark_llama)
            ]
            
            for api_name, benchmark_func in apis:
                print(f"  🔄 Running {api_name}...", end=" ", flush=True)
                result = benchmark_func(file_path, schema, schema_type)
                
                # Calculate accuracy if ground truth available
                if ground_truth and file_name in ground_truth and result.success:
                    result.accuracy_score = self.calculate_accuracy(
                        result.extracted_data,
                        ground_truth[file_name]
                    )
                
                self.results.append(result)
                
                if result.success:
                    print(f"✅ {result.execution_time:.2f}s | ${result.estimated_cost:.4f} | {result.accuracy_score:.1f}%")
                else:
                    print(f"❌ Failed: {result.error_message}")
        
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
    
    def generate_report(self, output_dir: str = "/mnt/user-data/outputs"):
        """Generate comprehensive benchmark report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. CSV with all results
        csv_path = output_path / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # 2. Summary statistics
        summary = df[df['success']].groupby('api_name').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'estimated_cost': ['mean', 'sum'],
            'accuracy_score': ['mean', 'std'],
            'success': 'count'
        }).round(4)
        
        summary_path = output_path / f"benchmark_summary_{timestamp}.csv"
        summary.to_csv(summary_path)
        
        # 3. JSON output with extracted data
        json_results = []
        for result in self.results:
            result_dict = asdict(result)
            json_results.append(result_dict)
        
        json_path = output_path / f"benchmark_detailed_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 4. Generate visualization report
        self._generate_visual_report(df, output_path, timestamp)
        
        return {
            'csv': str(csv_path),
            'summary': str(summary_path),
            'json': str(json_path)
        }
    
    def _generate_visual_report(self, df: pd.DataFrame, output_path: Path, timestamp: str):
        """Generate HTML report with visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        successful = df[df['success']]
        
        # 1. Execution Time Comparison
        sns.barplot(data=successful, x='api_name', y='execution_time', ax=axes[0, 0])
        axes[0, 0].set_title('Average Execution Time by API', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=15)
        
        # 2. Cost Comparison
        sns.barplot(data=successful, x='api_name', y='estimated_cost', ax=axes[0, 1])
        axes[0, 1].set_title('Average Cost by API', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Cost (USD)')
        axes[0, 1].tick_params(axis='x', rotation=15)
        
        # 3. Accuracy Comparison
        if 'accuracy_score' in successful.columns and successful['accuracy_score'].sum() > 0:
            sns.barplot(data=successful, x='api_name', y='accuracy_score', ax=axes[1, 0])
            axes[1, 0].set_title('Average Accuracy by API', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].tick_params(axis='x', rotation=15)
        
        # 4. Success Rate
        success_rate = df.groupby('api_name')['success'].mean() * 100
        success_rate.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Success Rate by API', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        chart_path = output_path / f"benchmark_charts_{timestamp}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OCR Benchmark Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #e8f5e9; border-radius: 8px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
        .success {{ color: #4CAF50; }}
        .failure {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 OCR Benchmark Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Summary Metrics</h2>
        <div>
"""
        
        # Add summary metrics
        for api in df['api_name'].unique():
            api_data = df[df['api_name'] == api]
            success_data = api_data[api_data['success']]
            
            if len(success_data) > 0:
                html_content += f"""
            <div class="metric">
                <div class="metric-label">{api}</div>
                <div class="metric-value">{success_data['execution_time'].mean():.2f}s</div>
                <div class="metric-label">Avg Time</div>
            </div>
            <div class="metric">
                <div class="metric-label">{api}</div>
                <div class="metric-value">${success_data['estimated_cost'].mean():.4f}</div>
                <div class="metric-label">Avg Cost</div>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>📈 Performance Charts</h2>
        <img src="{}" alt="Performance Charts">
        
        <h2>📋 Detailed Results</h2>
        <table>
            <tr>
                <th>API</th>
                <th>File</th>
                <th>Schema</th>
                <th>Time (s)</th>
                <th>Cost ($)</th>
                <th>Tokens</th>
                <th>Accuracy (%)</th>
                <th>Status</th>
            </tr>
""".format(chart_path.name)
        
        for _, row in df.iterrows():
            status = '✅ Success' if row['success'] else f"❌ {row['error_message']}"
            status_class = 'success' if row['success'] else 'failure'
            
            html_content += f"""
            <tr>
                <td>{row['api_name']}</td>
                <td>{row['file_name']}</td>
                <td>{row['schema_type']}</td>
                <td>{row['execution_time']:.2f}</td>
                <td>${row['estimated_cost']:.4f}</td>
                <td>{row['token_count']:,}</td>
                <td>{row['accuracy_score']:.1f}</td>
                <td class="{status_class}">{status}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        html_path = output_path / f"benchmark_report_{timestamp}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)


def main():
    """Main execution function"""
    import sys
    
    # API Keys - Replace with your actual keys
    AZURE_KEY = "your-azure-key"
    AZURE_ENDPOINT = "your-azure-endpoint"
    GEMINI_KEY = "your-gemini-key"
    LLAMA_KEY = "your-together-ai-key"
    
    # File paths
    INPUT_FILES = [
        "/mnt/user-data/uploads/1-7.pdf",
        "/mnt/user-data/uploads/Claim_Form_1.pdf",
        "/mnt/user-data/uploads/Claim_Form_2.pdf"
    ]
    
    SCHEMAS = {
        'invoice': "/mnt/user-data/uploads/schema.json",
        'claim': "/mnt/user-data/uploads/schema_claim.json"
    }
    
    # Optional: Ground truth data for accuracy measurement
    GROUND_TRUTH = {
        # Add ground truth data here if available
        # "1-7.pdf": { ... },
        # "Claim_Form_1.pdf": { ... },
    }
    
    # Initialize and run benchmark
    benchmark = OCRBenchmark(AZURE_KEY, AZURE_ENDPOINT, GEMINI_KEY, LLAMA_KEY)
    
    try:
        benchmark.run_benchmark(INPUT_FILES, SCHEMAS, GROUND_TRUTH)
        report_files = benchmark.generate_report()
        
        print("\n📊 Reports generated:")
        for report_type, path in report_files.items():
            print(f"  - {report_type}: {path}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
