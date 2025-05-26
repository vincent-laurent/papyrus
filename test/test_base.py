import pandas as pd
import pytest
from papyrus import engine
from papyrus.core.file import File

path = "./papyrus/data_pdf/invoice_100.pdf"


def run_extractor_tests(extractor):
    file = File(path)

    # Test extraction "all"
    file.extract(content="all", extractor=extractor)
    assert isinstance(file.text, str)
    assert isinstance(file.tables, list)


    # Test extraction "text"
    file = File(path)
    file.extract(content="text", extractor=extractor)
    assert isinstance(file.text, str)
    assert isinstance(file.tables, list)


    # Test extraction "tables"
    file = File(path)
    file.extract(content="tables", extractor=extractor)
    assert isinstance(file.text, str)
    assert isinstance(file.tables, list)



@pytest.mark.parametrize("extractor", [
    engine.PDFPlumberExtractor(),
    engine.DoclingExtractor(),
    engine.PyMuPDFExtractor(),
    engine.PyPDF2Extractor(),
    # engine.EasyOCRExtractor(),
    # engine.TesseractOCRExtractor(),
    # engine.HuggingFaceOCRExtractor(),
    engine.CamelotExtractor()
])
def test_all_extractors(extractor):
    run_extractor_tests(extractor)


