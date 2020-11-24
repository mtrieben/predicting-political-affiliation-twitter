import string

def replace_word(w):
    if w.startswith("http"):
        return "<LINK>"
    if w.startswith("#"):
        return w[1:]
    if w.startswith("@"):
        return w
    if w.startswith(".@"):
        return w[1:]
    return w.strip().strip(string.punctuation).lower()