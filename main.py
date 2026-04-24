from model.predict import predict
from nlp.interactive_nlp import interactive_nlp

def final_decision(cnn_conf, nlp_conf):
    return (cnn_conf * 0.6) + (nlp_conf * 0.4)


# TEST IMAGE
bitki, hastalik, cnn_conf = predict("test.jpg")

print("CNN sonucu:", hastalik, cnn_conf)

if hastalik == "septoria":

    print("NLP aşamasına geçiliyor...")

    nlp_conf = interactive_nlp()

    final = final_decision(cnn_conf, nlp_conf)

    print("FINAL SONUÇ:", final)