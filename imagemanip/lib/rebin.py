import numpy as n

def rebin(a, *args):
   shape = a.shape
   lenShape = len(shape)
   factor = n.asarray(shape)/n.asarray(args)
   evList = ['a.reshape('] + \
          ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
          [')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
   print ''.join(evList)
   return eval(''.join(evList))

