import warnings
from asl_data import SinglesData
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    x_len = test_set.get_all_Xlengths()
    for x,lengths in x_len.values():
      #logger.info("$s $s",x,lengths)
      scores, best_guess, highest_score = dict(), None, None
      for word, model in models.items():
        try:
          scores[word] = model.score(x, lengths)
          if highest_score is None or highest_score < scores[word]:
            highest_score, best_guess = scores[word], word
        except:
          scores[word] = None
      probabilities.append(scores)
      guesses.append(best_guess)

    return probabilities, guesses

