import stanza

stanza.download('en', processors='tokenize,ner')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
# nlp = stanza.Pipeline(lang='en', processors = {'tokenize': 'ewt','ner': 'ontonotes'})

txt_file = open('a9.txt', "r")
data = txt_file.read()
doc = nlp(data)

num_sent = 0
num_tok = 0
num_ents = 0
for sent in doc.sentences:
	num_sent += 1
	for token in sent.tokens:
		num_tok += 1
	for ent in sent.ents:
		if ent.type == "PERSON":
			num_ents += 1
# for ent in doc.ents:
# 	num_ents += 1

print(num_sent)
print(num_tok)
print(num_ents)
# print(*[f'token: {token.text}\tner: {token.ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')
# print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
# for sentence in doc.sentences:
# 	for word in sentence.words:
# 		print(word.text)

#Track listingEdit