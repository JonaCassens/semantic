import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp('cat')
word2 = nlp('monkey')
word3 = nlp('banana')
word4 = nlp('catnip')


print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print(word4.similarity(word1))

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go", "Hello, there is my car", "I\'ve lost my car in my car", "I\'d like my boat back", "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare) 
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#Words seem to be linked by categories and subcategories etc.
#Cat and Monkey are both animals which makes them very similar (category)
#Added dog to see if house pets would make it even more
#0.77 similarity between the two
#I assume each word has a large categorisation list (like a game of guess who) so for cat it would be, Animal, Pet, Fur, 4 Legs, etc.

#Using simpler model en_core_web_sm returns a warning saying that the model has no word vectors loaded so similiarity is less useful