from collections import Counter


def main():
    # - Gandalf the Grey
    txt = "It is not our part to master all the tides of the world, " \
          "but to do what is in us for the succor of those years wherein we are set, " \
          "uprooting the evil in the fields that we know, so that those who live after may have clean earth to till. " \
          "What weather they shall have is not ours to rule."

    corp = [[w.lower() for w in sntc.split()] for sntc in txt.split(".") if len(sntc) > 0]

    word_list = set()
    word_count = Counter()
    for sntc in corp:
        word_count += Counter(sntc)
        word_list.update(sntc)

    vocab_size = len(word_count)
    # Since we access the index of a word very often, we prebuild indices
    word_index = {w: i for i, w in enumerate(word_list)}
    word_index = {i: w for w, i in word_index.items()}





if __name__ == "__main__":
    main()