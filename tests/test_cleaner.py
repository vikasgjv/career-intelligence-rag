import unittest
from app.rag.cleaner import clean_text


class TestCleaner(unittest.TestCase):

    def test_unicode_and_bullet_cleaning(self):
        raw_text = "Alex Mercer\n• Python & SQL\nﬁrst line"
        cleaned = clean_text(raw_text)
        self.assertIn("- Python & SQL", cleaned)
        self.assertIn("first line", cleaned)

    def test_section_spacing(self):
        raw_text = "Alex Mercer\nSkills:\nPython\nProjects:\nCareer RAG"
        cleaned = clean_text(raw_text)
        self.assertIn("Skills:\nPython\n\nProjects:", cleaned)


if __name__ == "__main__":
    unittest.main()
