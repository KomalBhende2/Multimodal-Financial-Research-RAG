from pathlib import Path
import fitz


class PDFPageSplitter:

    def __init__(self, output_dir="data/page_cache"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_pdf(self, pdf_path):

        pdf_path = Path(pdf_path)

        doc = fitz.open(pdf_path)

        pages = []

        folder = self.output_dir / pdf_path.stem

        folder.mkdir(exist_ok=True)

        for page_number in range(len(doc)):

            new_pdf = fitz.open()

            new_pdf.insert_pdf(
                doc,
                from_page=page_number,
                to_page=page_number
            )

            page_file = folder / f"page_{page_number+1}.pdf"

            new_pdf.save(page_file)

            new_pdf.close()

            pages.append(page_file)

        doc.close()

        return pages