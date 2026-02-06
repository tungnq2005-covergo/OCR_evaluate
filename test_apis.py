#!/usr/bin/env python3
"""
Individual API Testing Script
Test each OCR API separately before running full benchmark
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure():
    """Test Azure Document Intelligence"""
    print("Testing Azure Document Intelligence")
    
    key = os.getenv('AZURE_KEY')
    endpoint = os.getenv('AZURE_ENDPOINT')
    
    if not key or not endpoint:
        print("❌ Azure credentials not found in .env")
        return False
    
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        # Test with first PDF
        test_file = "data/inputs/1-7.pdf"
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"📄 Analyzing: {test_file}")
        
        with open(test_file, 'rb') as f:
            poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=f,  # Pass file object directly
            content_type="application/pdf"
        )
        result = poller.result()
        
        print(f"✅ Success!")
        print(f"   Pages analyzed: {len(result.pages)}")
        print(f"   Tables found: {len(result.tables) if hasattr(result, 'tables') else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_gemini():
    """Test Google Gemini (New SDK: google-genai)"""
    print("Testing Google Gemini 2.0 Flash (New SDK)")
    
    key = os.getenv('GEMINI_API_KEY')
    
    if not key:
        print("❌ Gemini API key not found in .env")
        return False
    
    try:
        # Import thư viện SDK mới
        from google import genai
        from google.genai import types
        import time
        
        # 1. Khởi tạo Client (thay vì configure)
        client = genai.Client(api_key=key)
        
        # Test file check
        test_file = "data/inputs/1-7.pdf"
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            return False
        
        
        # 2. Upload file
        # Lưu ý: SDK mới dùng client.files.upload
        # Ta mở file ở chế độ binary 'rb' để an toàn nhất
        with open(test_file, "rb") as f:
            uploaded_file = client.files.upload(
                file=f,
                config=types.UploadFileConfig(mime_type="application/pdf")
            )
        

        # 3. Polling (Chờ xử lý)
        # SDK mới trả về object File, cần gọi get() lại để cập nhật trạng thái
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = client.files.get(name=uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            print("❌ File processing failed")
            return False
            
        print("   Generating content...")
        
        # 4. Generate Content
        # Dùng client.models.generate_content thay vì model.generate_content
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                uploaded_file, # Truyền trực tiếp object file vào
                "Extract all text from this document."
            ]
        )
        
        print(f"✅ Success!")
        print(f"   Response length: {len(response.text)} chars")
        
        # Usage metadata trong SDK mới truy cập hơi khác một chút (tùy version)
        if response.usage_metadata:
            print(f"   Tokens used: {response.usage_metadata.total_token_count}")
        
        # 5. Cleanup (Xóa file)
        # Dùng client.files.delete
        try:
            client.files.delete(name=uploaded_file.name)
            print("   Temporary file deleted.")
        except Exception as e:
            print(f"   Warning: Could not delete file: {e}")
        
        return True
        
    except ImportError:
        print("❌ google-genai package not installed or version mismatch")
        print("   Run: pip install google-genai")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        # In thêm chi tiết lỗi nếu cần debug
        # import traceback
        # traceback.print_exc()
        return False

def test_llama():
    """Test Llama Vision (LlamaCloud)"""
    print("Testing Llama Vision (LlamaCloud)")
    
    key = os.getenv('LLAMA_CLOUD_API_KEY')
    
    if not key:
        print("❌ LlamaCloud API key not found in .env")
        return False
    
    try:
        from llama_parse import LlamaParse
        
        # Initialize LlamaParse
        parser = LlamaParse(
            api_key=key,
            result_type="markdown",
            verbose=True
        )
        
        # Test with first PDF
        test_file = "data/inputs/1-7.pdf"
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            return False
    
        
        # Parse document
        documents = parser.load_data(test_file)
        
        if documents and len(documents) > 0:
            print(f"✅ Success!")
            print(f"   Pages parsed: {len(documents)}")
            print(f"   Content length: {len(documents[0].text)} chars")
            return True
        else:
            print("❌ No content extracted")
            return False
        
    except ImportError:
        print("❌ llama-parse package not installed")
        print("   Install with: pip install llama-parse")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test individual OCR APIs')
    parser.add_argument(
        '--api',
        choices=['azure', 'gemini', 'llama', 'all'],
        default='all',
        help='Which API to test (default: all)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("OCR API Testing Tool")
    print("="*60)
    
    # Check .env file
    if not Path('.env').exists():
        print("\n⚠️  .env file not found!")
        print("Please copy .env.example to .env and add your API keys.")
        return 1
    
    results = {}
    
    if args.api in ['azure', 'all']:
        results['azure'] = test_azure()
    
    if args.api in ['gemini', 'all']:
        results['gemini'] = test_gemini()
    
    if args.api in ['llama', 'all']:
        results['llama'] = test_llama()
    
    # Summary
    print("Test Summary")
    
    for api, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{api.upper():20s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All tests passed! Ready to run full benchmark.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check your API keys and configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
