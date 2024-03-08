def rebuild_text(tokens, trailing_whitespace):
    text = ""

    for token, ws in zip(tokens, trailing_whitespace):
        ws = " " if ws == True else ""
        text += token + ws

    return text
