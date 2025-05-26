# Copyright 2025 Mews Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
from abc import ABC, abstractmethod
from typing import Optional

from papyrus.core import file

from papyrus.tools.text_processing import text_without_tables


class BaseExtractor(ABC):
    """
    Abstract base class for all text extractors.
    All concrete extractors must implement the `_run_impl` method.
    """

    def __init__(self, capabilities=set()) -> None:
        self.capabilities = set()
        if not isinstance(self, BaseExtractor):
            raise TypeError(
                f"Expected extractor to be an instance of BaseExtractor, got {type(self).__name__} instead."
            )

    def run(self, file: "file.File") -> None:
        before = copy.deepcopy(file.pages)

        self._run_impl(file)

        if before == file.pages:
            raise RuntimeError("You have to modify 'file.pages' in your extractor.")

        for page_num, content in file.pages.items():
            if not isinstance(page_num, int):
                raise TypeError(f"Page key must be int, got {type(page_num)}")
            if not isinstance(content, dict):
                raise TypeError(
                    f"Page content must be dict, got {type(content)} for page {page_num}"
                )

            if "text" not in content or "tables" not in content:
                raise KeyError(
                    f"Page {page_num} dict must have keys 'text' and 'tables'"
                )
            if not isinstance(content["text"], str):
                raise TypeError(
                    f"Page {page_num} 'text' must be str, got {type(content['text'])}"
                )
            if not isinstance(content["tables"], list):
                raise TypeError(
                    f"Page {page_num} 'tables' must be list, got {type(content['tables'])}"
                )

    @abstractmethod
    def _run_impl(self, file: "file.File") -> None:
        """
        Run the extraction process and update the file.pages attribute.

        Args:
            file (File): The File object to update.
        """
        pass


class DoclingExtractor(BaseExtractor):
    def __init__(self, capabilities=set()):
        super().__init__()
        self.capabilities = {"text", "tables", "text_ocr", "tables_orc"}

    def _run_impl(self, file: "file.File"):
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError("'docling' is not installed. Run `pip install docling`")
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("'pandas' is not installed. Run `pip install pandas`")

        doc_converter = DocumentConverter()
        conv_res = doc_converter.convert(file.path)

        page_text = conv_res.document.export_to_text()
        page_md = conv_res.document.export_to_markdown()
        file.pages[0]["text"] = page_text
        file.pages[0]["text_md"] = page_md

        for table in conv_res.document.tables:
            table_df: pd.DataFrame = table.export_to_dataframe()
            file.pages[0]["tables"].append(table_df)


class PDFPlumberExtractor(BaseExtractor):
    """
    Extracts text and tables using the pdfplumber library.
    """

    def __init__(self, capabilities=set()):
        super().__init__()
        self.capabilities = {"text", "tables"}

    def _run_impl(self, file: "file.File") -> None:
        """
        Extract text and tables from the PDF using pdfplumber and update file.pages.

        Args:
            file (File): The File object to update.

        Raises:
            ImportError: If pdfplumber or pandas is not installed.
            FileNotFoundError: If the file path does not exist.
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "'pdfplumber' is not installed. Run `pip install pdfplumber`"
            )
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("'pandas' is not installed. Run `pip install pandas`")
        if not os.path.exists(file.path):
            raise FileNotFoundError(f"File not found: {file.path}")

        with pdfplumber.open(file.path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                page_text = text_without_tables(page) or ""
                file.pages[page_number]["text"] = page_text.strip()

                table_data = page.extract_table()
                if table_data:
                    df = pd.DataFrame(table_data)
                    if not df.empty:
                        file.pages[page_number]["tables"].append(df)


class PyMuPDFExtractor(BaseExtractor):
    """
    Extracts text and tables using the PyMuPDF (fitz) library.
    """

    def __init__(self, capabilities=set()):
        super().__init__()
        self.capabilities = {"text", "tables"}

    def _run_impl(self, file: "file.File") -> None:
        """
        Extract text and tables from the PDF using PyMuPDF and update file.pages.

        Args:
            file (File): The File object to update.

        Raises:
            ImportError: If PyMuPDF is not installed.
            FileNotFoundError: If the file path does not exist.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("'PyMuPDF' is not installed. Run `pip install pymupdf`")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("'pandas' is not installed. Run `pip install pandas`")

        if not os.path.exists(file.path):
            raise FileNotFoundError(f"File not found: {file.path}")

        doc = fitz.open(file.path)
        text = ""
        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text += page_text.strip()
            file.pages[page_number]["text"] = page_text.strip()

            tables = page.find_tables()
            for table in tables:
                df = table.to_pandas()
                if not df.empty:
                    file.pages[page_number]["tables"].append(df)


class PyPDF2Extractor(BaseExtractor):
    """
    Extracts text using the PyPDF2 library.
    """

    def __init__(self, capabilities=set()):
        super().__init__()
        self.capabilities = {"text"}

    def _run_impl(self, file: "file.File") -> None:
        """
        Extract text from the PDF using PyPDF2 and update file.pages.

        Args:
            file (File): The File object to update.

        Raises:
            ImportError: If PyPDF2 is not installed.
            FileNotFoundError: If the file path does not exist.
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("'PyPDF2' is not installed. Run `pip install PyPDF2`")

        if not os.path.exists(file.path):
            raise FileNotFoundError(f"File not found: {file.path}")

        with open(file.path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                file.pages[page_number]["text"] = text.strip()


class EasyOCRExtractor(BaseExtractor):
    """
    Extracts OCR text using the easyocr library.
    """

    def __init__(self, model: Optional[str] = "en"):
        self.model = model
        super().__init__()
        self.capabilities = {"text_ocr"}

    def _run_impl(self, file: "file.File") -> None:
        try:
            import easyocr
        except ImportError:
            raise ImportError("'easyocr' is not installed. Run `pip install easyocr`")

        if not os.path.exists(file.path):
            raise FileNotFoundError(f"File not found: {file.path}")

        reader = easyocr.Reader([self.model])
        results = reader.readtext(file.path, detail=0)

        file.pages[1]["text"] = "\n".join(results).strip()


class TesseractOCRExtractor(BaseExtractor):
    """
    Extracts OCR text using the Tesseract library.
    """

    def __init__(self, capabilities=set()):
        super().__init__()
        self.capabilities = {"text_ocr"}

    def _run_impl(self, file: "file.File") -> None:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError(
                "'pytesseract' or 'Pillow' is not installed. Run `pip install pytesseract Pillow`"
            )

        if not os.path.exists(file.path):
            raise FileNotFoundError(f"File not found: {file.path}")

        image = Image.open(file.path)
        ocr_text = pytesseract.image_to_string(image)
        file.pages[1]["text"] = ocr_text.strip()


class HuggingFaceOCRExtractor(BaseExtractor):
    """
    Extracts OCR text using a Hugging Face model (e.g., TroCR).
    """

    def __init__(
        self, model_name: str = "microsoft/trocr-base-printed", use_cuda: bool = False
    ):
        self.model_name = model_name
        self.use_cuda = use_cuda
        super().__init__()
        self.capabilities = {"text_ocr"}
        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForVision2Seq,
                BitsAndBytesConfig,
            )
        except ImportError:
            raise ImportError(
                "Missing required packages. Run `pip install torch transformers bitsandbytes`"
            )

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, quantization_config=quant_config
        )
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _run_impl(self, file: "file.File") -> None:
        if not os.path.exists(file.path):
            raise FileNotFoundError(f"File not found: {file.path}")

        try:
            import torch
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("Missing dependencies. Run `pip install torch pdf2image`")

        images = convert_from_path(file.path)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this image."}],
            }
        ]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for page_number, image in enumerate(images, start=1):
            try:
                image = image.convert("RGB")
                inputs = self.tokenizer(
                    text=[text_prompt], images=image, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(**inputs)

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                file.pages[page_number]["ocr_text"] = decoded.strip()

            except ValueError as e:
                print(f"Skipping page {page_number}: {e}")


class CamelotExtractor(BaseExtractor):
    def __init__(self, capabilities=set()):
        super().__init__()
        self.capabilities = {"text_ocr"}

    def _run_impl(self, file: "file.File"):
        try:
            import camelot
        except ImportError:
            raise ImportError("'camelot' is not installed. Run `pip install camelot`")

        tables = camelot.read_pdf(file.path, pages="all",flavor="stream")
        for page_number, table in enumerate(tables):
            file.pages[page_number]["tables"].append(table.df)
