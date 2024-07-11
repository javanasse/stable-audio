def get_custom_metadata(info, audio):

    prompt = info["relpath"]
    prompt = prompt.replace("_", " ").replace("/", " ")
    prompt = prompt.replace("/", " ")
    prompt = prompt.replace(".wav", "")
    
    # print(prompt)
    
    return {"prompt": prompt}