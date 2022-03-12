import numpy as np
import pandas as pd
from scipy import spatial


def main() -> None:
    vectors = {}
    with open("C:\\Users\\nikhi\\large_assets\\top_50000.txt", 'r', encoding='utf8') as glove:
        for line in glove:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            vectors[word] = vector
    
    def distance(word, reference):
        return spatial.distance.cosine(vectors[word], vectors[reference])
    
    def closest_words(reference):
        return sorted(vectors.keys(), key = lambda x: distance(x, reference))

    def goodness(word, answers, bad):
        if word in answers + bad: return -999
        return sum(distance(word, b) for b in bad) - 4.0 * sum(
            distance(word, g) for g in answers
        )
    
    def minimax(word, answers, bad):
        if word in answers + bad: return -999
        return min(distance(word, b) for b in bad) - max(
            distance(word, g) for g in answers
        )

    def candidates(answers, bad, size=100):
        best = sorted(vectors.keys(), key = lambda x: -1 * goodness(x, answers, bad))
        res = [(str(i + 1), "{0:.2f}".format(minimax(w, answers, bad)), w)
           for i, w in enumerate(sorted(best[:250], key=lambda w: -1 * minimax(w, answers, bad))[:size])]
        return [(". ".join([c[0], c[2]]) + " (" + c[1] + ")") for c in res]

    


    


if __name__ == "__main__":
    main()

