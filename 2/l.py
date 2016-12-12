from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math

def generate_data():
    classA = [(random.normalvariate(-2.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + \
        [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + \
        [(random.normalvariate(-1.5, 1), random.normalvariate(-2.5, 1), 1.0) for i in range(5)] + \
        [(random.normalvariate(1.5, 1), random.normalvariate(-2.5, 1), 1.0) for i in range(5)]
    classB = [(random.normalvariate(1.0, 0.5), random.normalvariate(-1.0, 0.5), -1.0) for i in range(10)] + \
        [(random.normalvariate(2.5, 0.5), random.normalvariate(2, 0.5), -1.0) for i in range(10)] + \
        [(random.normalvariate(-3.5, 0.5), random.normalvariate(-4, 0.5), -1.0) for i in range(10)]
    data = classA + classB
    random.shuffle(data)

    pylab.hold(True)
    pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')

    return data
#linear kernel
def linear_kernel(x1, x2):
    return numpy.dot(x1, x2) + 1

# nth degree polynomial kernel
def poly_kernel(x1, x2, degree):
    return pow(numpy.dot(x1, x2) + 1, degree)

# rbf kernel
def rbf_kernel(x1, x2, sigma):
    sub = [a[0] - a[1] for a in zip(x1, x2)]
    return math.exp(-(numpy.dot(sub, sub))/(2*pow(sigma,2)))

#sigmoid kernel
def sigmoid_kernel(x1, x2, k, delta):
    return math.tanh(k*numpy.dot(x1,x2) - delta)

def compute_kernel_matrix(data, kernel_function, *args, **keys):
    kmat = numpy.matrix(numpy.zeros((len(data), len(data))))
    for i in range(len(data)):
        for j in range(len(data)):
            # ignore the keys when passing to the linear kernel
            if (keys):
                kernel = kernel_function(data[i][:2], data[j][:2], **keys)
            else:
                kernel = kernel_function(data[i][:2], data[j][:2])
            kmat[i,j] = data[i][2]*data[j][2]*kernel
            
    return kmat

def indicator(x, y, support, kernel_function, *args, **keys):
    if (keys):
        return sum([s[1]*s[0][2]*kernel_function(s[0][:2], [x, y], **keys) for s in support])
    else:
        return sum([s[1]*s[0][2]*kernel_function(s[0][:2], [x, y]) for s in support])

def svm_compute(data):
    # define the function to use here
    kernel_function = rbf_kernel
    slack = True
    C = 2.0

    if (kernel_function == poly_kernel):
        params = dict(degree=2)
    elif (kernel_function == rbf_kernel):
        params = dict(sigma=15)
    elif (kernel_function == sigmoid_kernel):
        params = dict(k=2, delta=0.1)
    else:
        params = {}
    
    # if using anything other than linear kernel, more function arguments
    # are required to define the kernel. Pass one of these dicts to
    # the compute_kernel_matrix function and the indicator function
    K = matrix(compute_kernel_matrix(data, kernel_function, **params))
    q = matrix(-1.0, (len(data), 1))

    if (not slack):
        G = matrix(numpy.diag([-1.0]*len(data)))
        h = matrix(0.0, (len(data), 1))
    else:
        G = matrix(numpy.concatenate((numpy.diag([-1.0]*len(data)), numpy.diag([1.0]*len(data))), axis=0))
        h = matrix(numpy.concatenate(([0.0]*len(data), [1.0*C]*len(data)), axis=1))


    r = qp(K, q, G, h)
    alpha = list(r['x'])
    support = [(data[j], alpha[j]) for j in [i for i,v in enumerate(alpha) if v > 1e-5]]

    xrange = numpy.arange(-10,10, 0.05)
    yrange = numpy.arange(-10,10, 0.05)

    print(support)
    
    # don't forget to add the parameter dictionary to the indicator if needed.
    grid = matrix([[indicator(x, y, support, kernel_function, **params) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))
    
    pylab.show()
    
def main():
    data = generate_data()
    svm_compute(data)


if __name__ == '__main__':
    main()