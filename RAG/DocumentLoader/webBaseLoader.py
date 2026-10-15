from langchain_community.document_loaders import WebBaseLoader

url = 'https://docs.langchain.com/oss/python/releases/changelog'

loader = WebBaseLoader(url)

print(loader.load())