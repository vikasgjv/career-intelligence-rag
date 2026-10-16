import unittest
import tempfile
from io import BytesIO
from pathlib import Path
from app.rag.loader import load_document, load_pdf, load_text, load_pdf_pages, load_document_pages


def create_minimal_pdf_bytes(text: str = "Alex Mercer - Python Developer") -> bytes:
    """Generates a valid, minimal 1-page PDF byte string in pure Python without external dependencies."""
    stream_content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET\n".encode("latin1")
    stream_len = len(stream_content)

    obj1 = b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    obj2 = b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    obj3 = b"3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <</Font <</F1 4 0 R>>>> /MediaBox [0 0 612 792] /Contents 5 0 R>>\nendobj\n"
    obj4 = b"4 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n"
    obj5 = f"5 0 obj\n<</Length {stream_len}>>\nstream\n".encode("latin1") + stream_content + b"endstream\nendobj\n"

    header = b"%PDF-1.4\n"
    offset1 = len(header)
    offset2 = offset1 + len(obj1)
    offset3 = offset2 + len(obj2)
    offset4 = offset3 + len(obj3)
    offset5 = offset4 + len(obj4)
    startxref = offset5 + len(obj5)

    xref = (
        f"xref\n0 6\n0000000000 65535 f \n{offset1:010d} 00000 n \n{offset2:010d} 00000 n \n{offset3:010d} 00000 n \n{offset4:010d} 00000 n \n{offset5:010d} 00000 n \n"
    ).encode("latin1")

    trailer = f"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n{startxref}\n%%EOF\n".encode("latin1")

    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref + trailer


class TestLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_dir_path = Path(cls.temp_dir.name)

        cls.txt_path = cls.temp_dir_path / "resume.txt"
        cls.txt_path.write_text("Alex Mercer\nSkills:\nPython, SQL, Machine Learning\n", encoding="utf-8")

        cls.pdf_bytes = create_minimal_pdf_bytes("Alex Mercer - Python Developer")
        cls.pdf_path = cls.temp_dir_path / "resume.pdf"
        cls.pdf_path.write_bytes(cls.pdf_bytes)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_load_text_file(self):
        text = load_document(str(self.txt_path))
        self.assertIsInstance(text, str)
        self.assertIn("Alex Mercer", text)
        self.assertIn("Python", text)

    def test_load_pdf_file(self):
        text = load_document(str(self.pdf_path))
        self.assertIsInstance(text, str)
        self.assertIn("Alex Mercer", text)
        self.assertIn("Python", text)

    def test_load_pdf_pages_metadata(self):
        pages = load_pdf_pages(str(self.pdf_path))
        self.assertIsInstance(pages, list)
        self.assertGreaterEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Alex Mercer", pages[0]["text"])

    def test_load_pdf_pages_from_bytes(self):
        pdf_bytes = BytesIO(self.pdf_bytes)
        pages = load_pdf_pages(pdf_bytes)
        self.assertIsInstance(pages, list)
        self.assertGreaterEqual(len(pages), 1)
        self.assertIn("Alex Mercer", pages[0]["text"])

    def test_corrupt_pdf_raises_value_error(self):
        corrupt_bytes = BytesIO(b"not a valid pdf content")
        with self.assertRaises(ValueError):
            load_pdf_pages(corrupt_bytes)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_document(str(self.temp_dir_path / "non_existent.pdf"))

    def test_unsupported_extension(self):
        dummy_path = self.temp_dir_path / "test_dummy.docx"
        dummy_path.touch()
        try:
            with self.assertRaises(ValueError):
                load_document(str(dummy_path))
        finally:
            if dummy_path.exists():
                dummy_path.unlink()


if __name__ == "__main__":
    unittest.main()


