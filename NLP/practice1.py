def usr_input():
    para = input("Enter the para : ")
    return(para)
    
def remove_pun(para):
    para_split = para.split(" ")
    con = []
    pun = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', 
    ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    for word in para_split:
        while word and word[-1] in pun:
            word = word[:-1]
        con.append(word)
        
    para_mods = " ".join(con)
    return(para_mods)

def main():
    print("Choose input type : \n 1. For manual \n 2. For Auto")
    type = int(input("Enter : "))
    
    
    if type == 1:
        user = usr_input()
    else : 
        file_name = "NLP/TextFiles/para_split.txt"
        file_ = open(file_name,"r")
        user = file_.read()
        
        
    clean_inp = remove_pun(user)
    print(clean_inp)


if __name__ == "__main__":
    main()