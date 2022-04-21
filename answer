#Samarth Ganesh Thopaiah, Jacqueline Liao
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
    # QA_MODEL=os.environ.get('QA_MODEL_2', 'deepset/roberta-base-squad2')
    QA_MODEL_ROBERTA=os.environ.get('QA_MODEL_2', 'deepset/roberta-base-squad2') #CHANGED
    QA_MODEL_BERT=os.environ.get('deepset/minilm-uncased-squad2', "deepset/minilm-uncased-squad2") #CHANGED

    #Initializaion of modules

    #See modules.py for each of these classes
    # document_retriever=DocumentRetrieval()
    query_cleaner=QueryProcessor(nlp)
    passage_retriever = PassageRetrieval(nlp)
    # answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)
    answer_extractor_roberta = AnswerExtractor(QA_MODEL_ROBERTA, QA_MODEL_ROBERTA) #CHANGED
    answer_extractor_bert = AnswerExtractor(QA_MODEL_BERT, QA_MODEL_BERT) #CHANGED

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

        # answers = answer_extractor.extract(question, passages)
        # all_answers=[]
        # # print("Question: {}".format(str(question)))
        # # print('Potential Answers Pipeline 1 Offline')
        # # for i in answers:
        # #     print("{},{}".format(i['score'],i['answer']))
        # for i in answers:
        #     all_answers.append(i)
        # all_answers.sort(key=operator.itemgetter('score'), reverse=True)
        # print("A{} {}".format(counter,all_answers[0]['answer']))

        # CHANGED
        roberta_scores, roberta_answers = answer_extractor_roberta.extract(question, passages) 
        bert_scores, bert_answers = answer_extractor_bert.extract(question, passages) 

        roberta_answers = [query_cleaner.generate_query(a) for a in roberta_answers]
        bert_answers = [query_cleaner.generate_query(a) for a in bert_answers]


        overlap_answers = [value for value in roberta_answers if value in bert_answers]
        answer_probs = []

        if len(overlap_answers) != 0:
          answer_probs = []
          for a in overlap_answers:
            roberta_index = roberta_answers.index(a)
            bert_index = bert_answers.index(a)
            roberta_p = roberta_scores[roberta_index]
            bert_p = bert_scores[bert_index]
            answer_probs.append((roberta_p+bert_p)/2)

          max_value = max(answer_probs)
          max_index = answer_probs.index(max_value)
          best_ans =  overlap_answers[max_index]
        else:
          best_ans = None
          best_score = 0
          for i in range(len(roberta_answers)):
            if roberta_scores[i] > best_score:
              best_ans = roberta_answers[i]
              best_score = roberta_scores[i]

        print("A{} {}".format(counter,best_ans)) #CHANGED
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

        
