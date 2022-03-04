import spacy
import os
from spacy.scorer import Scorer
from datasets import load_dataset
from spacy.training import offsets_to_biluo_tags
from spacy.tokens import Doc
import io

#sPacy transformer models roBERTa

#Evaluation Tasks
    # Sentence segmentation
    # Tokenization
    # Named entity recognition

#Evaluation 
    #Precision 
    #Recall
    #F1-Score

#Named Entity Recognition

#conll2003


data_oracle=load_dataset("conll2003",split='validation')

def get_tok_tag(file):
    token=[]
    ner=[]
    for i in file:
        temp=i.split(' ')
        if (temp[0]==('-DOCSTART-')) or temp[0]=="\n" or temp[0] =='':
            pass
        else:
            token.append(temp[0])
            ner.append(temp[-1])
    return token,ner

token,ner=get_tok_tag(data_oracle['text'][2:100])

print("Number of NER tags {}".format(len(ner)))
print("Number of Tokens {}".format(len(token)))
assert(len(ner)==len(token))
ground_truth=[]
for i in ner: 
    if i =='B-PER' or i == 'I-PER':
        ground_truth.append(i)
    else: 
        ground_truth.append('O')

print(ground_truth)

# #Validation Dataset

# #We convert list of tokens to string

str_token=''.join(token)
#print(token)

nlp = spacy.load("en_core_web_trf")
print(nlp.pipe_names)
# doc = nlp(Doc(nlp.vocab, words=token))
doc=nlp(str_token)
test=[]
tags=[]
for entity in doc.ents:
    print(entity.label_)
    # temp=[(entity.start_char,entity.end_char, entity.label_)]
    # tags.append(offsets_to_biluo_tags(doc,temp))
    
# print(tags)

# #Converting BILUO to BIO tags 
def biluo_to_bio(input):
    tags=[element for sublist in input for element in sublist]
    bio_tags=[]
    for i in tags: 
        if (i==('B-PERSON')):
            temp='B-PER'
        elif (i==('L-PERSON')):
            temp='I-PER'
        elif (i==('U-PERSON')):
            temp='B-PER'
        else:
            temp=i
        bio_tags.append(temp)
    return bio_tags

bio_tags=biluo_to_bio(tags)
# print(bio_tags)
print('Number of ground truth tags {}'.format(len(ground_truth)))
print('Number of BIO tags {}'.format(len(bio_tags)))


# #We just report simply accuracy since we are just looking at PER. 



# length=len(bio_tags)
# accuracy=0
# for i in range(length):
#     if (bio_tags[i]==ground_truth[i]):
#         accuracy+=1
# print('Accuracy is {}%'.format((accuracy/length)*100))



#Project Dataset 

data_lst=[]
path="C:\\Users\\thopa\\Desktop\\Assignments\\11-611 NLP\\Project\\nlp-project-dev-data-articles\\nlp-project-dev-data-articles\\set2"

nlp = spacy.load("en_core_web_trf")
#doc = nlp(str(flattened_list))
counter=0
entity_counter=0
without_line=0
for i in os.listdir(path):
    temp=i
    if (temp.split('.')[1]=='txt'):
        print('File name: {} '.format(i))
        f=io.open(path +'\\'+ i,mode='r',encoding='utf-8')
        data_project=f.read()
        doc = nlp(str(data_project))

        with open('terminal_output.txt', 'w') as f:
            #Same loop used for NER, token extraction and sentence seg
            for entity in doc.ents:
                #print(entity.text)
                # temp=entity.text
                # entity_counter+=1

                # if (temp =='\n'):
                #     without_line+=0
                # else:
                #     without_line+=1
                # # else: 
                # #     pass

                # if(entity.label_=='PERSON'):
                counter+=1
                print(entity.text, entity.label_)
            


print('NER Person count {}'.format(counter))
#print('Sentence  count {}'.format(entity_counter))
#print('Without line counter {}'.format(without_line))

      