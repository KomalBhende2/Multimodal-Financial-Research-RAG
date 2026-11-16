from registry.document_registry import DocumentRegistry

registry = DocumentRegistry()

documents = registry.process_documents()

print("\n")

for doc in documents:

    print(doc)