from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

import logging
from typing import Dict, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class BSHTMLLoaderCustom(BaseLoader):
    """Loader that uses beautiful soup to parse HTML files."""

    def __init__(
        self,
        # file_path: str,
        html_doc: str,
        page_url: str,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object."""
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        # self.file_path = file_path
        self.html_doc = html_doc 
        self.page_url = page_url
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def load(self) -> List[Document]:
        from bs4 import BeautifulSoup

        """Load HTML document into document objects."""
        # with open(self.file_path, "r", encoding=self.open_encoding) as f:
            # soup = BeautifulSoup(f, **self.bs_kwargs)
        soup = BeautifulSoup(self.html_doc, **self.bs_kwargs)

        text = soup.get_text(self.get_text_separator)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": self.page_url,
            "title": title,
        }
        return [Document(page_content=text, metadata=metadata)]
