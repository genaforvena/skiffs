import unittest
from summarize import divide_text


class TestCode(unittest.TestCase):
    text = "On. Somehow on. Till nohow oh. Bigger that the. cut window or it abrupltyly cuts the last sentence! Fin"

    def test_divide_cuts_into_sentences(self):
        chunks = divide_text(self.text, 3)

        print(str(chunks))
        expected = [
            "On. Somehow on.",
            "Till nohow oh.",
            "Bigger that the.",
            "cut window",
            "or it",
            "abrupltyly cuts",
            "the last sentence!",
            "Fin",
        ]
        self.assertEqual(chunks, expected)


if __name__ == "__main__":
    unittest.main()
