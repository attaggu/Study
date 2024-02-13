









# def Name(*names):
#     for name in names:
#         print("%s %s" % (name[0],name[1:3]), end='' )
#         print("\n")
#         print(type(names),names)
         
# Name('이태규', '김상헌', '이정훈')

# # 이 태규
# # <class 'tuple'> ('이태규', '김상헌', '이정훈')

# # 김 상헌
# # <class 'tuple'> ('이태규', '김상헌', '이정훈')

# # 이 정훈
# # <class 'tuple'> ('이태규', '김상헌', '이정훈')

# def Name(**kwargs):
#     for key, value in kwargs.items():
#         print("{0} am {1}".format(key, value))

# Name(I='taegyu lee')
# # I am taegyu lee

# def Name(**kwargs):
#     for key, value in kwargs.items():
#         if 'answer' in kwargs.keys():
#             print("nice to meet you")
#         else:print("{0} am {1}".format(key, value))
#         print(type(kwargs),kwargs)
        
# Name(I = 'taegyu lee')
# Name(answer = 'taegyu')

# # I am taegyu lee
# # <class 'dict'> {'I': 'taegyu lee'}
# # nice to meet you
# # <class 'dict'> {'answer': 'taegyu'}

