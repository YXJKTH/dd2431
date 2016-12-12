import monkdata as m
import dtree as d

import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]
'''
# assignment 1
print '*** assignment 1 ***'
print 'entropy of monk-1 dataset: ' + str(d.entropy(m.monk1))
print 'entropy of monk-2 dataset: ' + str(d.entropy(m.monk2))
print 'entropy of monk-3 dataset: ' + str(d.entropy(m.monk3))

# assignment 2
print '\n*** assignment 2 ***'
print '-- information gain of monk-1 dataset: --'
print 'a_1: ' + str(d.averageGain(m.monk1, m.attributes[0]))
print 'a_2: ' + str(d.averageGain(m.monk1, m.attributes[1]))
print 'a_3: ' + str(d.averageGain(m.monk1, m.attributes[2]))
print 'a_4: ' + str(d.averageGain(m.monk1, m.attributes[3]))
print 'a_5: ' + str(d.averageGain(m.monk1, m.attributes[4]))
print 'a_6: ' + str(d.averageGain(m.monk1, m.attributes[5]))
print '-- information gain of monk-2 dataset: --'
print 'a_1: ' + str(d.averageGain(m.monk2, m.attributes[0]))
print 'a_2: ' + str(d.averageGain(m.monk2, m.attributes[1]))
print 'a_3: ' + str(d.averageGain(m.monk2, m.attributes[2]))
print 'a_4: ' + str(d.averageGain(m.monk2, m.attributes[3]))
print 'a_5: ' + str(d.averageGain(m.monk2, m.attributes[4]))
print 'a_6: ' + str(d.averageGain(m.monk2, m.attributes[5]))
print '-- information gain of monk-3 dataset: --'
print 'a_1: ' + str(d.averageGain(m.monk3, m.attributes[0]))
print 'a_2: ' + str(d.averageGain(m.monk3, m.attributes[1]))
print 'a_3: ' + str(d.averageGain(m.monk3, m.attributes[2]))
print 'a_4: ' + str(d.averageGain(m.monk3, m.attributes[3]))
print 'a_5: ' + str(d.averageGain(m.monk3, m.attributes[4]))
print 'a_6: ' + str(d.averageGain(m.monk3, m.attributes[5]))

# assignment 3
print '\n*** assignment 3 ***'
print '-- generated tree for monk-1 dataset: --'
t = d.buildTree(m.monk1, m.attributes)
print(t)
print 'test set error: ' + str(d.check(t, m.monk1test))
print 'train set error: ' + str(d.check(t, m.monk1))
print '-- generated tree for monk-2 dataset: --'
t = d.buildTree(m.monk2, m.attributes)
print(t)
print 'test set error: ' + str(d.check(t, m.monk2test))
print 'train set error: ' + str(d.check(t, m.monk2))
print '-- generated tree for monk-3 dataset: --'
t = d.buildTree(m.monk3, m.attributes)
print(t)
print 'test set error: ' + str(d.check(t, m.monk3test))
print 'train set error: ' + str(d.check(t, m.monk3))
'''
# assignment 4
print '\n*** assignment 4 ***'
train, val = partition(m.monk3, 0.8)
flag = True
t = d.buildTree(train, m.attributes) # generate a tree

while flag:
    print '-- new tree generated --'
    flag = False
    err_test = d.check(t, val) # unpruned test set error
    print 'train set error: ' + str(err_test)
    for foo in d.allPruned(t):
        err_pruned = d.check(foo, val)
        if err_pruned > err_test:
            print 'pruned solution found with error: ' + str(err_pruned)
            t = foo
            flag = True

print '-- no pruned solution found. --'