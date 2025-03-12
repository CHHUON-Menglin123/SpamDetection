import unittest
from naive_bayes_spam import predict_spam

class TestSpamDetection(unittest.TestCase):
    def test_spam_prediction(self):
        self.assertEqual(predict_spam("Congratulations, you have won a lottery!"), 'spam')
        self.assertEqual(predict_spam("Hi, can we meet tomorrow?"), 'not spam')

if __name__ == '__main__':
    unittest.main()
