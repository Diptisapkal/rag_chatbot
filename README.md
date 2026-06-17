# RAG Chatbot — PDF वर प्रश्न-उत्तर

PDF upload kara आणि त्यातील content वर आधारित प्रश्न विचारा. हा project RAG (Retrieval-Augmented Generation) architecture वापरतो — म्हणजे LLM फक्त त्याच्या training knowledge वर अवलंबून न राहता, तुमच्या दिलेल्या document मधून answer काढतो.

## कसं काम करतं (Architecture)

1. **PDF load + chunk** — मोठी PDF छोट्या तुकड्यांत (chunks) तोडली जाते
2. **Embeddings** — प्रत्येक chunk चं vector representation तयार होतं (free, local Hugging Face model वापरून)
3. **Vector store** — सगळे embeddings ChromaDB मध्ये साठवले जातात
4. **Retrieval** — user च्या प्रश्नाशी सर्वात जवळचे chunks शोधले जातात
5. **Generation** — ते chunks + प्रश्न LLM ला (Groq चा Llama 3.3 70B — मोफत आणि खूप fast) दिले जातात, आणि उत्तर तयार होतं

## Setup

```bash
pip install -r requirements.txt
```

**Note:** LangChain ची नवीन version खूप वारंवार internal structure बदलते. जर अजूनही कोणतीही `ModuleNotFoundError` आली, तर ती कोणत्या package मधून यायला पाहिजे ते शोधण्यासाठी `pip show <package-name>` वापरून बघ, किंवा मला error paste करून सांग — मी import line लगेच fix करून देईन.

**Groq API key** हवी (पूर्ण मोफत, credit card लागत नाही):
1. https://console.groq.com वर जा
2. Sign up कर (Google/email ने)
3. डाव्या sidebar मध्ये "API Keys" वर क्लिक कर
4. "Create API Key" बटण दाब, नाव दे, आणि key copy कर
5. App च्या sidebar मध्ये ती key paste कर

embeddings साठी कोणतीही key लागत नाही (free, local Hugging Face model वापरलंय).

## Run करायचं

```bash
streamlit run app.py
```

ब्राउझरमध्ये आपोआप उघडेल (साधारण `localhost:8501`).

## Portfolio साठी हे project कसं present कराल

- GitHub repo च्या README मध्ये एक screen-recording GIF टाका (app चालताना दिसेल)
- "What I learned" section लिहा — chunking strategy, embeddings, vector search याबद्दल थोडं
- एक 1-2 ओळींचं technical summary लिहा: *"Built a RAG pipeline using LangChain, ChromaDB, Hugging Face embeddings, and Groq's Llama 3.3 for fast, free LLM inference to enable Q&A over custom PDF documents."* — हे resume किंवा LinkedIn वर वापरता येईल

## पुढे काय सुधारणा करता येईल (Advanced — दाखवण्यासाठी छान)

- **Multiple PDFs** एकत्र upload करायची सोय
- **Streaming responses** — उत्तर शब्द-शब्द दिसावं (ChatGroq मध्ये streaming=True)
- **Persistent vector store** — सध्या in-memory आहे, ते disk वर साठवायचं (Chroma चं persist_directory वापरून)
- **Deployment** — Streamlit Community Cloud वर मोफत deploy करता येतं (live link मिळतो, जो resume मध्ये टाकता येतो)

## सामान्य अडचणी (Troubleshooting)

- **`ModuleNotFoundError`** — `pip install -r requirements.txt` पुन्हा run करा
- **Embeddings download मध्ये वेळ लागतोय** — पहिल्यांदा Hugging Face model download होतो (~90MB), एकदाच होतं
- **Rate limit / 429 error** — Groq चं free tier मिनिटाला ठराविक requests देतं, थोडं थांबून पुन्हा प्रयत्न कर
