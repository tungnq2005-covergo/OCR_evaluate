# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Or run the setup script:
```bash
./setup.sh
```

### Step 2: Configure API Keys

Edit the `.env` file (created from `.env.example`):

```env
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-azure-key-here
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Google Gemini  
GOOGLE_GEMINI_API_KEY=your-gemini-key-here

# Together AI (Llama)
TOGETHER_API_KEY=your-together-key-here
```

### Step 3: Run Benchmark

Test individual APIs first (recommended):
```bash
python test_apis.py
```

Run full benchmark:
```bash
python run_benchmark.py
```

## 📊 What You'll Get

After running, check `/mnt/user-data/outputs/` for:

1. **CSV Report** - Detailed results in spreadsheet format
2. **Summary Statistics** - Aggregated metrics by API
3. **JSON Data** - Full extraction results
4. **HTML Report** - Interactive visualization
5. **Charts** - Performance comparison graphs

## 🔍 Understanding Results

### Metrics Explained

**Execution Time**
- Time to process document (seconds)
- Lower is better
- Includes upload, processing, and extraction

**Cost**
- Estimated cost in USD
- Based on current API pricing
- Lower is better for budget-conscious users

**Accuracy**
- Percentage match with ground truth (0-100%)
- Only available if ground truth provided
- Higher is better

**Success Rate**
- Percentage of successful extractions
- Should be 100% for production use

### Typical Results

| API | Speed | Cost | Accuracy | Best For |
|-----|-------|------|----------|----------|
| Azure | Fast | Medium | High | Simple documents, speed priority |
| Gemini | Medium | Low | High | Complex layouts, balanced needs |
| Llama | Medium | Low | Good | Cost optimization, flexibility |

## 🛠️ Troubleshooting

### Test API Connection
```bash
python test_apis.py --api azure    # Test Azure only
python test_apis.py --api gemini   # Test Gemini only
python test_apis.py --api llama    # Test Llama only
python test_apis.py --api all      # Test all (default)
```

### Common Issues

**"Missing API key"**
→ Check `.env` file has correct keys

**"File not found"**
→ Ensure PDFs are in `/mnt/user-data/uploads/`

**"Rate limit exceeded"**
→ Wait a few minutes, or add delays in code

**"Invalid response"**
→ Check API quotas and account status

## 📁 Project Files

```
.
├── ocr_benchmark.py              # Main benchmark engine
├── run_benchmark.py              # Simple runner
├── test_apis.py                  # API testing tool
├── requirements.txt              # Dependencies
├── .env.example                  # Config template
├── setup.sh                      # Setup script
├── ground_truth_template.json    # Accuracy measurement
└── README.md                     # Full documentation
```

## 🎯 Next Steps

1. **Customize schemas** - Edit `schema.json` or `schema_claim.json`
2. **Add ground truth** - Fill in `ground_truth_template.json`
3. **Test more files** - Add PDFs to benchmark
4. **Optimize costs** - Compare APIs for your use case
5. **Automate** - Integrate into your workflow

## 💡 Pro Tips

- **Test First**: Always run `test_apis.py` before full benchmark
- **Ground Truth**: Accuracy measurement requires correct ground truth data
- **Cost Control**: Start with small batches to estimate costs
- **Schema Design**: Well-designed schemas improve extraction accuracy
- **Error Handling**: Check `error_message` field in results for failures

## 📞 Need Help?

1. Check the full **README.md** for detailed documentation
2. Review API provider documentation
3. Verify API keys and quotas
4. Look at example ground truth data
5. Check execution logs for error details

## ⚡ Quick Commands

```bash
# Setup
./setup.sh

# Test APIs
python test_apis.py

# Run benchmark
python run_benchmark.py

# View results
ls -lh /mnt/user-data/outputs/
```

---

Ready to benchmark? Run `python test_apis.py` first! 🚀
