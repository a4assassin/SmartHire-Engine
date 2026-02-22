from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#PDFloader = PyMuPDFLoader("Sample_CV3.pdf")
PDFloader = PyMuPDFLoader("CV_ML_2025_v3.pdf")

doc = PDFloader.load()

text = "\n".join(doc[i].page_content for i in range (len(doc)))

schema = [
    ResponseSchema(name = "professional_summary", description = "Summary of the candidate's professional background often described under heading Profile Summary, About me, Professional Summary etc. in the CV." ),
    ResponseSchema(name = "work_experience", description = "A list of the candidate's work experiences, including job titles, durations, and key responsibilities or achievements for each role."),
    ResponseSchema(name = "education", description = "A list of the candidate's educational qualifications, including only degree names, and graduation years, if found."),
    ResponseSchema(name = "projects", description = "If found, a list of the candidate's projects, including project names, skills used and key responsibilities or achievements for each project, if found."),
    ResponseSchema(name = "skills", description = "A list of the candidate's skills, including both technical and soft skills, if found."),
    ResponseSchema(name = "certifications", description = "A list of the candidate's certifications, including certification names, if found.")
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template='Below is the CV of a candidate and you have to extract the details as it is:\n{CV_Text} \n {format_instruction}',
    input_variables=['CV_Text'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model | parser
res = chain.invoke(text)
print(res)