import unittest

from mmrag.retrieval.service import extract_reference_numbers, infer_source_types


class RetrievalTests(unittest.TestCase):
    def test_source_type_inference(self) -> None:
        self.assertEqual(infer_source_types("README 里怎么安装？"), ["doc"])
        self.assertEqual(infer_source_types("Issue 里有哪些 bug？"), ["issue"])
        self.assertEqual(infer_source_types("最近哪个 PR 合并了？"), ["pr"])

    def test_reference_number_extraction(self) -> None:
        self.assertEqual(extract_reference_numbers("issue #12 和 PR #34 有什么关系？"), [12, 34])


if __name__ == "__main__":
    unittest.main()
