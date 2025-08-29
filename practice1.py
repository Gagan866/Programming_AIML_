# kap_num = input("Enter a number : ")
# count = 0
# if (len(kap_num) != 4 or len(set(kap_num))==1):
#     print("Enter number with 4 digits and tow unique digits ")
    
# else:
#     kap_num = int(kap_num)
#     print("Number : ",kap_num)
#     while kap_num != 6174:
#         num_str = str(kap_num).zfill(4)
#         print("Number after filling 0 : ",num_str)
#         asc = int("".join(sorted(num_str)))
#         dec = int("".join(sorted(num_str,reverse=True)))
#         kap_num = dec-asc
#         count += 1
#         print(f"{count}  {dec} - {asc} = {kap_num}")
#     print("Kaprekars num : ",kap_num)    
    
    
def kapekar(num):
    new =  int(''.join(map(str, num)))  # list to int
    iteration = 0
    while new != 6174 :
        new = list(str(new))  #int to list
        max_n = form_max_num(new)  # form largest number
        min_n = form_min_num(new)  # form smallest number
        new =  diff_(max_n,min_n)
        iteration += 1
        print("Iteration:",iteration)
        print( max_n,min_n,new)
        
          

def  form_max_num(num):
    num.sort(reverse=True)
    max_num = int(''.join(map(str, num)))
    print("max number:",max_num)
    return max_num

def form_min_num(num):
    min_num = sorted(num)
    if min_num[0] == '0' and min_num[1] == '0':
        i, j = 0, 2
        min_num[i], min_num[j] = min_num[j], min_num[i]
    elif min_num[0] == '0':
         i, j = 0, 1
         min_num[i], min_num[j] = min_num[j], min_num[i]
    min_num = int(''.join(map(str, min_num)))
    print("min number:",min_num)
    return min_num

    
def diff_(m,n):
    return(m-n)


num = list(input("Enter four digits:"))
if len(num) == 4 and len(set(num))>=2 :
    print(num)
    kapekar(num)
else:
    print(" enter four digits , with atleast uniques")