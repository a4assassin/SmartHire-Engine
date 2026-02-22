from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
PDFloader = PyMuPDFLoader("Sample_CV3.pdf")

doc = PDFloader.load()

text = "\n".join(doc[i].page_content for i in range (len(doc)))
print(text)
