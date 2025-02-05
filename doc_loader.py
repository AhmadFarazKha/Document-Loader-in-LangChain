import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredHTMLLoader, CSVLoader

# 1. Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file")

# 2. Initialize Gemini Chat Model
chat = ChatGoogleGenerativeAI(model="gemini-pro")

# 3. Define the data directory and supported file types
data_dir = "./data"
supported_files = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".csv": CSVLoader  # Added CSV support
}

print("Processing files in data directory:\n")

# 4. Process each file in the data directory
for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext in supported_files:
        try:
            # Load appropriate document loader
            loader_class = supported_files[file_ext]
            loader = loader_class(filepath)
            documents = loader.load()

            print(f"\n=== Contents of {filename} ===")
            for doc in documents:
                print(doc.page_content + "\n")

            # Generate summary using Gemini
            combined_content = "\n".join([doc.page_content for doc in documents])
            summary_prompt = f"Please summarize the following {file_ext.upper()} document:\n\n{combined_content}"
            summary = chat.invoke(summary_prompt)
            
            print(f"\n=== Summary of {filename} ===")
            print(summary.content + "\n")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    else:
        print(f"Skipping unsupported file: {filename}")

print("\nProcessing complete!")
