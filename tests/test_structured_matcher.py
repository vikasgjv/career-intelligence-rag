import unittest
from app.rag.structured_matcher import evaluate_structured_requirement, detect_structured_requirement_type


class TestStructuredMatcher(unittest.TestCase):

    def test_cgpa_strong_match(self):
        req_item = {"requirement": "Minimum Academic Score: CGPA 6.0 / 60% and above", "category": "eligibility"}
        resume_text = "Education: Bachelor of Engineering CGPA - 8.63"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "STRONG EVIDENCE")
        self.assertEqual(res["match_method"], "structured_numeric")
        self.assertEqual(res["candidate_value"], 8.63)
        self.assertEqual(res["required_value"], 6.0)

    def test_cgpa_formatting_variations(self):
        req_item = {"requirement": "Required CGPA: 7.0 or above", "category": "eligibility"}

        for resume_sample in [
            "CGPA of 8.63 in Computer Science",
            "Cumulative GPA: 8.63 / 10.0",
            "8.63 / 10 CGPA",
            "Grade Point Average: 8.63"
        ]:
            res = evaluate_structured_requirement(req_item, resume_sample)
            self.assertIsNotNone(res, f"Failed on {resume_sample}")
            self.assertEqual(res["status"], "STRONG EVIDENCE", f"Failed on {resume_sample}")
            self.assertEqual(res["candidate_value"], 8.63)

    def test_higher_required_cgpa(self):
        req_item = {"requirement": "Minimum CGPA 9.0", "category": "eligibility"}
        resume_text = "Education: Bachelor of Engineering CGPA - 8.63"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "MISSING / UNVERIFIED")
        self.assertEqual(res["candidate_value"], 8.63)
        self.assertEqual(res["required_value"], 9.0)

    def test_missing_cgpa(self):
        req_item = {"requirement": "Minimum CGPA 6.0", "category": "eligibility"}
        resume_text = "Education: Bachelor of Engineering in Computer Science"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "MISSING / UNVERIFIED")
        self.assertIsNone(res["candidate_value"])

    def test_percentage_strong_and_missing_match(self):
        req_item = {"requirement": "Minimum 60% and above", "category": "eligibility"}
        resume_high = "Education: Passed with 78% aggregate."
        res = evaluate_structured_requirement(req_item, resume_high)
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "STRONG EVIDENCE")
        self.assertEqual(res["match_method"], "structured_numeric")

        resume_low = "Education: 55% in graduation."
        res_low = evaluate_structured_requirement(req_item, resume_low)
        self.assertEqual(res_low["status"], "MISSING / UNVERIFIED")

        resume_none = "Education: Bachelor of Science."
        res_none = evaluate_structured_requirement(req_item, resume_none)
        self.assertEqual(res_none["status"], "MISSING / UNVERIFIED")
        self.assertIsNone(res_none["candidate_value"])

    def test_backlog_strong_match(self):
        req_item = {"requirement": "No Active Backlogs", "category": "eligibility"}
        resume_text = "Active Backlogs: 0"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "STRONG EVIDENCE")
        self.assertEqual(res["match_method"], "structured_eligibility")

    def test_missing_backlog_info_does_not_assume_zero(self):
        req_item = {"requirement": "No Active Backlogs", "category": "eligibility"}
        resume_text = "Bachelor of Engineering in Artificial Intelligence"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "MISSING / UNVERIFIED")
        self.assertIsNone(res["candidate_value"])

    def test_graduation_year_strong_match(self):
        req_item = {"requirement": "2027 graduating batch only", "category": "eligibility"}
        resume_text = "Presidency University, Bengaluru 03-2023 - 05-2027"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "STRONG EVIDENCE")
        self.assertEqual(res["match_method"], "structured_eligibility")

    def test_unrelated_year_does_not_infer_graduation_year(self):
        req_item = {"requirement": "2027 graduating batch only", "category": "eligibility"}
        resume_text = "Projects: Built web application in 2021."
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "MISSING / UNVERIFIED")
        self.assertIsNone(res["candidate_value"])

    def test_degree_strong_match(self):
        req_item = {"requirement": "Degree: BE / B.Tech / M.Tech / ME / MCA", "category": "education"}
        resume_text = "Education: Bachelor of Engineering: Artificial Intelligence and Machine Learning"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "STRONG EVIDENCE")
        self.assertEqual(res["match_method"], "structured_degree")

    def test_branch_strong_and_missing_match(self):
        req_item = {"requirement": "Branch: CS / IT / ECE", "category": "eligibility"}
        resume_text = "Education: Bachelor of Technology in Computer Science and Engineering"
        res = evaluate_structured_requirement(req_item, resume_text)

        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "STRONG EVIDENCE")
        self.assertEqual(res["match_method"], "structured_branch")

        resume_mech = "Education: Bachelor of Technology in Mechanical Engineering"
        res_mech = evaluate_structured_requirement(req_item, resume_mech)
        self.assertEqual(res_mech["status"], "MISSING / UNVERIFIED")

    def test_experience_structured_match(self):
        req_campus = {"requirement": "Campus hiring", "category": "experience"}
        resume_student = "Education: Bachelor of Engineering, Expected 2025"
        res = evaluate_structured_requirement(req_campus, resume_student)
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "STRONG EVIDENCE")
        self.assertEqual(res["match_method"], "structured_experience")

        req_years = {"requirement": "2+ years of experience", "category": "experience"}
        resume_exp = "Experience: 3 years of experience as Python Developer"
        res_exp = evaluate_structured_requirement(req_years, resume_exp)
        self.assertIsNotNone(res_exp)
        self.assertEqual(res_exp["status"], "STRONG EVIDENCE")


if __name__ == "__main__":
    unittest.main()
