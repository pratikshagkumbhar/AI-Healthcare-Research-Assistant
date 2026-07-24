from document_loader import load_and_split_pdf

pdf_path = "uploads/sample.pdf"

chunks = load_and_split_pdf(pdf_path)

print("Number of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content[:500])