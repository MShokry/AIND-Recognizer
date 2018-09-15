import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import logging
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def base_model_cv(self, num_states,X,lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_score = None
        min_model = None
        try:
            for n in range(self.min_n_components,self.max_n_components+1):
                model = self.base_model(n)
                logL = model.score(self.X,self.lengths)
                p = n ** 2 + 2 * n * model.n_features - 1 # p is the number of parameters
                logN = np.log(len(self.X)) #N is the number of data points
                score = -2 * logL + p * logN
                if min_score is None or min_score > score:
                    min_model,min_score = model , score
        except Exception as e:
            print(e)
            pass

        return min_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        max_score,best_model = None,None
        all_but_w = list(self.words)
        all_but_w.remove(self.this_word)
        #logger.info("Len of words %s",len(all_but_w))
        for n in range(self.min_n_components,self.max_n_components+1):
            model = self.base_model(n)
            try:
                score_self = model.score(self.X,self.lengths)
            except :
                continue
            score_but_i = 0.0
            for w in all_but_w:
                x,lengths = self.hwords[w]
                score_but_i += model.score(x,lengths)
            total_score = score_self - ( score_but_i / (len(self.words) - 1))
            if max_score is None or max_score < total_score:
                max_score,best_model = total_score,model

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select2(self):
        mean_scores = []
        # Save reference to 'KFold' in variable as shown in notebook
        split_method = KFold()
        try:
            for n_component in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n_component)
                # Fold and calculate model mean scores
                fold_scores = []
                for _, test_idx in split_method.split(self.sequences):
                    # Get test sequences
                    test_X, test_length = combine_sequences(test_idx, self.sequences)
                    # Record each model score
                    fold_scores.append(model.score(test_X, test_length))

                # Compute mean of all fold scores
                mean_scores.append(np.mean(fold_scores))
        except Exception as e:
            pass

        num_components = range(self.min_n_components, self.max_n_components + 1)
        states = num_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        return self.base_model(states)    
    
    def selectms(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        average,best_model,best_model_num = None, None, None
        for n in range(self.min_n_components,self.max_n_components+1):
            scores = []
            split_method = KFold(2 if len(self.sequences)<3 else 3)
            try:
                #logger.info("Number of components word %s " ,n)
                for train_idx , test_idx in split_method.split(self.sequences):
                    #logger.info("Train fold indices:{} Test fold indices:{}".format(train_idx, test_idx))  # view indices of the folds
                    X,lengths = combine_sequences(train_idx,self.sequences)
                    model_train = self.base_model_cv(n,X,lengths)
                    X,lengths = combine_sequences(test_idx,self.sequences)
                    try:
                        score = model_train.score(X,lengths)
                        scores.append(score)
                    except:
                        continue
                avg = np.mean(scores)
            except Exception as e:
                continue
            #logger.info("avg %s",avg)
            if average is None or average < avg:
                average, best_model_num = avg,n
        best_model = self.base_model(best_model_num)
        return best_model

    def select(self):
        best_cv_score = float('-inf')
        best_hmm_model = None
        kf = KFold()
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            log_sum = 0
            counter = 0
            try:
                for cv_train_ix, cv_test_ix in kf.split(self.sequences):
                    X_train, lengths_train = combine_sequences(cv_train_ix, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_ix, self.sequences)
                    try:
                        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        logL = hmm_model.score(X_test, lengths_test)
                        counter += 1
                    except:
                        logL = 0
                    log_sum += logL
                # Calculate Score
                cv_score = log_sum / (1 if counter == 0 else counter)

                # Select best model
                if cv_score > best_cv_score:
                    best_cv_score = cv_score
                    best_hmm_model = hmm_model
            except Exception as e:
                continue
        return best_hmm_model