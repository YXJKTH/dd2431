import monkdata as m
import dtree as dtree

foo = dtree.select(m.monk1, m.attributes[4], 3)
print '-- information gain of monk-1 dataset: --'
print 'a_1: ' + str(dtree.averageGain(foo, m.attributes[0]))
print 'a_2: ' + str(dtree.averageGain(foo, m.attributes[1]))
print 'a_3: ' + str(dtree.averageGain(foo, m.attributes[2]))
print 'a_4: ' + str(dtree.averageGain(foo, m.attributes[3]))
print 'a_6: ' + str(dtree.averageGain(foo, m.attributes[5]))

foo = dtree.select(m.monk1, m.attributes[4], 1)
print '-- is a_5 with value = 1 a majority class? --'
print dtree.mostCommon(foo)