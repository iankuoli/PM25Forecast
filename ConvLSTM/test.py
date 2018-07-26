from reader import construct_time_map2

aaa = [1,2,3,4,5,6,7,8,9,10,11,12,13]
ss = [(4,1), (10,2)]
ttt = construct_time_map2(aaa, ss)
yyy = aaa[ss[-1][0]:]
print(ttt)
print(yyy)
