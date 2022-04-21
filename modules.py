from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline
import operator
from gensim.summarization.bm25 import BM25
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline
import wikipedia
import requests
import concurrent.futures
import re




#Pre-Processing 
class QueryProcessor:

    def __init__(self, nlp, keep=None):
        self.nlp = nlp
        self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}

    def generate_query(self, text):
        proper_noun=0
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == 'PROPN':
                proper_noun+=1
        if proper_noun>=1:
            query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
        else:
            query = ' '.join(token.text for token in doc)
        return query

class DocumentRetrieval:

    def __init__(self, url='https://en.wikipedia.org/w/api.php'):
        self.url = url

    def search_pages(self, query):
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json'
        }
        res = requests.get(self.url, params=params)
        return res.json()

    def search_page(self, page_id):
        res = wikipedia.page(pageid=page_id)
        return res.content

    def search(self, query):
        pages = self.search_pages(query)

        #We keep the top 2 documents 
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            process_list = [executor.submit(self.search_page, page['pageid']) for page in pages['query']['search']]
            process_list=process_list[0:3]
            docs = [self.post_process(p.result()) for p in process_list]
        return docs

    def post_process(self, doc):
        pattern = '|'.join([
            '== References ==',
            '== Further reading ==',
            '== External links',
            '== See also ==',
            '== Sources ==',
            '== Notes ==',
            '== Further references ==',
            '== Footnotes ==',
            '=== Notes ===',
            '=== Sources ===',
            '=== Citations ===',
        ])
        
        p = re.compile(pattern)
        indices = [m.start() for m in p.finditer(doc)]
        if indices == []:
            return 'No content'
        else:
            min_idx = min(*indices, len(doc))
            return doc[:min_idx]


class PassageRetrieval:

    def __init__(self, nlp,keep=None):
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
        self.bm25 = None
        self.passages = []
        self.nlp = nlp
        self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}

    def preprocess(self, doc):
        passages = [p for p in doc.split('\n') if p and not p.startswith('=')]

        return passages
    
    def generate_query(self, text):
        doc = self.nlp(text)
        query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
        return query

    def fit(self, docs):
        # corpus=[]
        # for doc in docs:
        docs=self.generate_query(docs)
        #     passages = self.preprocess(doc)
        #     corpus=[self.tokenize(p) for p in passages]
        #     self.passages=passages
        #     print(len(self.passages))
        #     self.bm25 = BM25(corpus)
        passages = self.preprocess(docs)
        corpus = [self.tokenize(p) for p in passages]
        self.bm25 = BM25(corpus)
        self.passages = passages


    def most_similar(self, question, topn=3):
        tokens = self.tokenize(question)
        scores = self.bm25.get_scores(tokens)
        pairs = [(s, i) for i, s in enumerate(scores)]
        pairs.sort(reverse=True)
        passages = [self.passages[i] for _, i in pairs[:topn]]
        return passages

class AnswerExtractor:

    def __init__(self, tokenizer, model):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    # def extract(self, question, passages):
    #     answers = []
    #     for passage in passages:
    #         try:
    #             answer = self.nlp(question=question, context=passage)
    #             answer['text'] = passage
    #             answers.append(answer)
    #         except KeyError:
    #             pass
    #     answers.sort(key=operator.itemgetter('score'), reverse=True)
    #     return answers

    def extract(self, question, passages): #CHANGED
        all_scores = []
        all_as = []
        for passage in passages:
            try:
                answers = self.nlp(question=question, context=passage, top_k=3) 
                for answer in answers: 
                  all_scores.append(answer['score'])
                  all_as.append(answer['answer'])
            except KeyError:
                pass
        # answers.sort(key=operator.itemgetter('score'), reverse=True)
        return (all_scores, all_as)



     
    

    
