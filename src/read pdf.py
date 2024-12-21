from pypdf import PdfReader
import spacy
import spacy.cli
#spacy.cli.download("en_core_web_sm")


#function to get text from pdf
def read_pdf(pdf_name):
    reader = PdfReader(pdf_name)
    page = reader.pages[0]
    text = page.extract_text()
    return text
text = read_pdf("Sample PDFs/Sample Document (2).pdf")

import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)