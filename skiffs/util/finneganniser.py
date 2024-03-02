import random



# Function to transform a sentence into a Finnegans Wake style sentence
def finnegannise(text):
    words = text.split()

    transformed_words = []
    for word in words:
        if random.random() < 0.85:
            continue

        if len(word) > 3:
            new_word = word[:2] + random.choice(["ish", "esque", "ian"]) + word[2:]
        else:
            new_word = word
        if random.random() < 0.3:
            new_word = "".join(random.sample(new_word, len(new_word)))


        transformed_words.append(new_word)

    transformed_text = " ".join(transformed_words)
    return transformed_text

