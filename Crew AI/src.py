import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from crewai_tools import LlamaIndexTool

# Nom du fichier PDF local
local_filename = "Uber-Annual-Report-2019.pdf"

# Vérifier si le fichier existe
if not os.path.exists(local_filename):
    print(f"Erreur : Le fichier {local_filename} n'existe pas dans le répertoire courant.")
    exit()

# Créer un dossier temporaire pour stocker le PDF
temp_dir = "temp_pdf_dir"
os.makedirs(temp_dir, exist_ok=True)

# Copier le PDF dans le dossier temporaire
import shutil
shutil.copy(local_filename, os.path.join(temp_dir, local_filename))

# Utiliser SimpleDirectoryReader pour lire le PDF
print(f"Lecture du contenu du PDF : {local_filename}")
reader = SimpleDirectoryReader(temp_dir)
documents = reader.load_data()

# Afficher le contenu du premier document (il n'y en a qu'un dans ce cas)
print("Contenu extrait (1000 premiers caractères) :")
print(documents[0].text[:1000])

# Nettoyer : supprimer le dossier temporaire et son contenu
print("Nettoyage des fichiers temporaires...")
for file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file))
os.rmdir(temp_dir)
print("Nettoyage terminé.")


print(documents[1].get_content())


llm = OpenAI(model="gpt-4o")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)

# try out query engine tool

query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Uber 2019 10K Query Tool",
    description="Use this tool to lookup the 2019 Uber 10K Annual Report",
)

print(query_tool.args_schema.schema())