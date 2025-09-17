import os
from dotenv import load_dotenv
from Bio import Entrez #special tools from biopython to access NCBI databases
from Bio import Medline #to parse medline format

EMAIL = os.getenv("NCBI_EMAIL") # Always tell NCBI who you are
SEARCH_TERM = "Alzheimer's disease treatment" # Example search term
MAX_PAPERS = 50 # Maximum number of papers to fetch

Entrez.email = EMAIL # Always tell NCBI who you are
DATA_DIR = "Alzheimer_data" # Directory to save fetched data
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Step 1: Search for papers
handle = Entrez.esearch(db="pubmed", term=SEARCH_TERM, retmax=MAX_PAPERS) # Search PubMed
record = Entrez.read(handle) # Read the search results
id_list = record["IdList"] # List of PubMed IDs

# Step 2: Fetch details for each paper
handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text") # Fetch details
records = Medline.parse(handle)

# Step 3: Save details to files
saved_count = 0
for record in records:
    print(record)  # Print the entire record for debugging
    if "TI" in record and "AB" in record:
        title = record["TI"]
        abstract = record["AB"]

        filename = os.path.join(DATA_DIR, f"{record['PMID']}.txt")
        with open(filename, "w" , encoding="utf-8") as f:
            f.write(f"Title: {title}\n\nAbstract: {abstract}") # Correct

        saved_count += 1

print(f"Successfully saved {saved_count} papers to the '{DATA_DIR}' directory.")