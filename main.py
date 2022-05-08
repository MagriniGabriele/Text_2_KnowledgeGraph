import spacy
import neuralcoref

if __name__ == '__main__':

    #Caricamento modello inglese web per spacy
    nlp = spacy.load('en_core_web_sm')

    #Settaggio NeuralCoref
    neuralcoref.add_to_pipe(nlp)

    #Testo di esempio
    text = "Rihanna is basically master of the fashion universe right now, so we're naturally going to pay attention " \
           "to what trends she is and isn't wearing whenever she steps out of the door (or black SUV). She's having " \
           "quite the epic week, first presenting her Savage x Fenty lingerie runway show then hosting her annual " \
           "Diamond Ball charity event last night. Rihanna was decked out in Givenchy for the big event, " \
           "but upon arrival at the venue, she wore a T-shirt, diamonds (naturally), and a scarf, leather pants, " \
           "and heels in fall's biggest color trend: pistachio green. "
    doct = nlp(text)

    #Risoluzione coreferenza
    resolved_doc = doct._.coref_resolved
    print(resolved_doc)
    text1 = nlp(resolved_doc)

    #Named entity recognition COREFERENZA NON RISOLTA
    for ent in doct.ents:
        print(f"Named Entity '{ent.text}' with label '{ent.label_}'")

    print("\n")

    #Named entity recognition COREFERENZA RISOLTA
    for word in text1.ents:
        print(f"Named Entity '{word.text}' with label '{word.label_}'")
