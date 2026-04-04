#Takes all PDFs from data/raw/ and extract clean text, saves to data/parsed/

import pathlib
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend  # ← add this
from config import DATA_RAW, DATA_PARSED

def parse_all():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend  # ← add this
            )
        }
    )
    out_dir = pathlib.Path(DATA_PARSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(pathlib.Path(DATA_RAW).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found. Add PDFs first.")
        return []
    
    parsed = []
    
    for pdf_path in pdfs:
        print(f"Parsing: {pdf_path.name}...")
        try:
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()

            doc = {"source":pdf_path.name, "text":text}
            out_path = out_dir/(pdf_path.stem + ".json")
            out_path.write_text(json.dumps(doc, indent=2))
            parsed.append(doc)
            print(f"OK — {len(text):,} characters")
        
        except Exception as e:
            print(f"Failed: {pdf_path.name} - {e}")

    print(f"\nParsed {len(parsed)}/{len(pdfs)} PDFs → {DATA_PARSED}/")
    return parsed
    
if __name__=="__main__":
    parse_all()
