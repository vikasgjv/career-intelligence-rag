import unittest
from app.rag.requirements import extract_requirements


class TestRequirements(unittest.TestCase):

    def get_req_names(self, reqs):
        return [r["requirement"] if isinstance(r, dict) else r for r in reqs]

    def test_prototype_jd_extraction(self):
        jd = """
        Software Engineer Intern
        We are looking for a Software Engineer Intern with experience in:
        Python, SQL, Machine Learning
        Responsibilities:
        Develop machine learning applications.
        """
        reqs = self.get_req_names(extract_requirements(jd))
        self.assertIn("Python", reqs)
        self.assertIn("SQL", reqs)
        self.assertIn("Machine Learning", reqs)

    def test_combined_skills_splitting(self):
        jd = """
        Technical Requirements:
        - Programming in Java, C++, Python, or Node.js
        - Experience with Git and GitHub
        """
        reqs = extract_requirements(jd)
        req_names = self.get_req_names(reqs)

        self.assertIn("Java", req_names)
        self.assertIn("C++", req_names)
        self.assertIn("Python", req_names)
        self.assertIn("Node.js", req_names)
        self.assertIn("Git", req_names)
        self.assertIn("GitHub", req_names)

        # Verify traceability to authentic source sentence
        py_req = next(r for r in reqs if r["requirement"] == "Python")
        self.assertEqual(py_req["source_section"], "Technical Requirements:")
        self.assertIn("Java, C++, Python", py_req["source_text"])

    def test_company_marketing_exclusion(self):
        jd = """
        Your Role and Responsibilities:
        - Python
        - SQL

        About Acme Corp:
        We are a global provider of cutting-edge tech. Founded in 1995, headquartered in San Francisco.
        We pride ourselves on culture and equal opportunity employer principles.
        All qualified applicants will receive consideration without regard to race, religion, or gender.
        """
        reqs = extract_requirements(jd)
        req_names = self.get_req_names(reqs)

        self.assertIn("Python", req_names)
        self.assertIn("SQL", req_names)

        # Verify marketing and legal boilerplate are excluded
        for name in req_names:
            self.assertNotIn("global provider", name.lower())
            self.assertNotIn("equal opportunity", name.lower())
            self.assertNotIn("founded in", name.lower())

    def test_eligibility_and_education_extraction(self):
        jd = """
        Eligibility Criteria:
        Degree: BE / B.Tech / M.Tech / ME / MCA
        Branch: CS / IT / ECE / EEE
        2027 graduating Batch only
        Minimum Academic Score: CGPA 6.0 / 60% and above
        No Active Backlogs
        """
        reqs = extract_requirements(jd)
        req_names = self.get_req_names(reqs)

        self.assertTrue(any("degree" in r.lower() or "be" in r.lower() for r in req_names))
        self.assertTrue(any("2027" in r for r in req_names))
        self.assertTrue(any("cgpa" in r.lower() or "60%" in r for r in req_names))
        self.assertIn("No Active Backlogs", req_names)

    def test_responsibility_fine_grained_splitting(self):
        jd = """
        Your Role and Responsibilities:
        As a Software Developer you'll participate in many aspects of the software development lifecycle, such as design, code implementation, testing, and support.
        Analyze, validate and propose improvements to existing failures.
        """
        reqs = extract_requirements(jd)
        req_names = self.get_req_names(reqs)

        self.assertIn("Software Development Life Cycle Concepts", req_names)
        self.assertIn("Software Design", req_names)
        self.assertIn("Software Testing", req_names)
        self.assertIn("Debugging & Issue Analysis", req_names)
        self.assertIn("Solution Improvement", req_names)

    def test_soft_skills_extraction(self):
        jd = """
        Required Skills:
        - Problem solving
        - Communication skills
        - Interpersonal skills
        """
        reqs = extract_requirements(jd)
        req_names = self.get_req_names(reqs)

        self.assertIn("Problem Solving", req_names)
        self.assertIn("Communication Skills", req_names)
        self.assertIn("Interpersonal Skills", req_names)


if __name__ == "__main__":
    unittest.main()

