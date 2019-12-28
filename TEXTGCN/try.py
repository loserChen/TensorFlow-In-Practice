


# dataset='20ng'
# doc_name_list=[]
# f = open('data/' + dataset + '.txt', 'r')
# lines = f.readlines()
# for line in lines:
#     doc_name_list.append(line.strip())
#     print(line.strip())
train_ids=[1,2,3,4]
train_ids_str = '\n'.join(str(index) for index in train_ids)
print(train_ids_str)
