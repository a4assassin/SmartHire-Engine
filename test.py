from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="SEMANTIC_SIMILARITY")

#PDFloader = PyMuPDFLoader("Sample_CV3.pdf")
PDFloader = PyMuPDFLoader("C:/Users/a4anu/Downloads/CV_ML_2025_v3.pdf")

doc = PDFloader.load()

#text = "\n".join(doc[i].page_content for i in range (len(doc)))

CV_schema = [
    ResponseSchema(name = "professional_summary", description = "Summary of the candidate's professional background often described under heading Profile Summary, About me, Professional Summary etc. in the CV." ),
    ResponseSchema(name = "work_experience", description = "A list of the candidate's work experiences, including job titles, durations, and key responsibilities or achievements for each role."),
    ResponseSchema(name = "projects", description = "If found, a list of the candidate's projects, including project names, skills used and key responsibilities or achievements for each project, if found."),
    ResponseSchema(name = "skills", description = "A list of the candidate's skills, including both technical and soft skills, if found."),
    ResponseSchema(name = "education", description = "A list of the candidate's educational qualifications, including only degree names, and graduation years, if found."),
    ResponseSchema(name = "certifications", description = "A list of the candidate's certifications, including certification names, if found.")
]

JD_schema = [
    ResponseSchema(name = "responsibilities", description = "Key responsibilities or expected duties mentioned, not secondary or optional responsibilities/skills."),
    ResponseSchema(name = "minimum_experience", description = "Experience required for the job, if found."),
    ResponseSchema(name = "preferred_skills", description = "Preferred or required techinical stack/skills if mentioned"),
    ResponseSchema(name = "nicetohave_skills", description = "Not mandatory skills but often mentioned as nice to have or optional skills, if any."),
    ResponseSchema(name = "qualifications", description = "Preferred Educational Qualifications if mentioned.")
]

CV_parser = StructuredOutputParser.from_response_schemas(CV_schema)
JD_parser = StructuredOutputParser.from_response_schemas(JD_schema)


CV_template = PromptTemplate(
    template='Below is the CV of a candidate and you have to extract the details as it is:\n{CV_Text} \n {format_instruction}',
    input_variables=['CV_Text'],
    partial_variables={'format_instruction':CV_parser.get_format_instructions()}
)

JD_template = PromptTemplate(
    template='Below is the job description and you have to extract the details as it is:\n{JD_Text} \n {format_instruction}',
    input_variables=['JD_Text'],
    partial_variables={'format_instruction':JD_parser.get_format_instructions()}
)
JD = open("JD.txt", "r").read()

'''
chain = template | model | parser
res = chain.invoke(text)

chain2 = JD_template | model | JD_parser
res2 = chain2.invoke(JD)

for i, key in enumerate(res2):
    print(key,"\n", res2[key], "\n\n")

'''

'''
embed = []
for key in res:
    embed.append(embeddings.embed_documents([res[key]]))
'''