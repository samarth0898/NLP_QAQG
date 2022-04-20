#Samarth Ganesh Thopaiah
## 11-611 Natural Language Processing, Carnegie Mellon University

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np
import warnings
import spacy
import os
import operator
from modules import QueryProcessor,PassageRetrieval,AnswerExtractor,DocumentRetrieval
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import argparse
from pathlib import Path

def main(root_dataset,question_txt):
    #Language models

    #For pre-processing both query and selected document 
    spacy_model='en_core_web_lg'

    nlp = spacy.load(spacy_model, disable=['ner', 'parser', 'textcat'])
    QA_MODEL=os.environ.get('QA_MODEL_2', 'deepset/roberta-base-squad2')

    #Initializaion of modules

    #See modules.py for each of these classes
    # document_retriever=DocumentRetrieval()
    query_cleaner=QueryProcessor(nlp)
    passage_retriever = PassageRetrieval(nlp)
    answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)

    # root_dataset='set3\\a1.txt'
    # question_txt='questions.txt'
    with open(Path(root_dataset), "r",encoding='UTF-8') as f:
        article=f.read()

    with open(Path(question_txt), "r",encoding='UTF-8') as f:
        questions=f.read()
    questions_lst=questions.split("\n")
    counter=1
    for question in questions_lst:
        # query = query_cleaner.generate_query(question)
        # retriever=document_retriever.search(query)
        # retriever.append(article)

        #Pipe 1
        passage_retriever.fit(article)
        passages = passage_retriever.most_similar(question)
        answers = answer_extractor.extract(question, passages)
        all_answers=[]
        # print("Question: {}".format(str(question)))
        # print('Potential Answers Pipeline 1 Offline')
        # for i in answers:
        #     print("{},{}".format(i['score'],i['answer']))
        for i in answers:
            all_answers.append(i)
        all_answers.sort(key=operator.itemgetter('score'), reverse=True)
        print("A{} {}".format(counter,all_answers[0]['answer']))
        counter+=1

def argparser():
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument("article_path",help="path to article.txt")
    parser.add_argument("question_path",help="path to questions.txt")
    return(parser)



if __name__=="__main__":
    

    parser=argparser()
    args = parser.parse_args()
    main(root_dataset=args.article_path,question_txt=args.question_path)

        
