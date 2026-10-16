import unittest
from app.rag.chunker import chunk_text


class TestChunker(unittest.TestCase):

    def test_section_aware_chunking(self):
        text = "Alex Mercer\n\nSkills:\nPython, SQL, Machine Learning\n\nProjects:\nCareer Intelligence RAG\n\nEducation:\nBachelor's degree"
        chunks = chunk_text(text)

        self.assertGreaterEqual(len(chunks), 3)
        sections = [c["section"] for c in chunks]
        self.assertIn("Skills", sections)
        self.assertIn("Projects", sections)
        self.assertIn("Education", sections)

    def test_page_number_metadata_preservation(self):
        pages = [
            {"page_number": 1, "text": "Skills:\nPython, FastAPI"},
            {"page_number": 2, "text": "Education:\nBachelor of Science\n\nExperience:\nSoftware Engineer at Acme"}
        ]
        chunks = chunk_text(pages)
        self.assertGreater(len(chunks), 0)

        page_1_chunks = [c for c in chunks if c.get("page") == 1]
        page_2_chunks = [c for c in chunks if c.get("page") == 2]

        self.assertGreaterEqual(len(page_1_chunks), 1)
        self.assertGreaterEqual(len(page_2_chunks), 2)
        self.assertTrue(all("section" in c and "source" in c for c in chunks))

    def test_fine_grained_skill_decomposition(self):
        text = "Skills:\nPython, SQL, Machine Learning"
        chunks = chunk_text(text)
        texts = [c["text"] for c in chunks]
        self.assertIn("Python", texts)
        self.assertIn("SQL", texts)
        self.assertIn("Machine Learning", texts)

    def test_named_project_chunking(self):
        text = """
Projects:
Career Intelligence RAG:
Building a Retrieval-Augmented Generation system.

Network Traffic Classification:
Developed an ML model using Python.
"""
        chunks = chunk_text(text)
        project_chunks = [c for c in chunks if c["section"] == "Projects"]
        self.assertGreaterEqual(len(project_chunks), 2)

    def test_empty_input_chunking(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text([]), [])


if __name__ == "__main__":
    unittest.main()

