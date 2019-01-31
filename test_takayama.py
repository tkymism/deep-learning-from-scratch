import numpy as np
import matplotlib.pylab as plt
import unittest
import perceptron as pct

class NumpyTestGroup(unittest.TestCase):

    def test_np_case001(self):
        x = np.array([1.0,2.0,3.0])
        y = np.array([2.0,3.0,6.0])
        z = x + y
        self.assertEqual(z[0], 3.0)
        self.assertEqual(z[1], 5.0)
        self.assertEqual(z[2], 9.0)

    def test_np_case002(self):
        x = np.array([1.0,2.0,3.0])
        y = np.array([2.0,3.0,6.0])
        z = x * y
        self.assertEqual(z[0],  2.0)
        self.assertEqual(z[1],  6.0)
        self.assertEqual(z[2], 18.0)

    def test_np_case003(self):
        x = np.array([1.0,2.0,3.0])
        z = -1 * x
        self.assertEqual(z[0], -1.0)
        self.assertEqual(z[1], -2.0)
        self.assertEqual(z[2], -3.0)
        e = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(e[0,0], 1.0)
        self.assertEqual(e[0,1], 2.0)
        self.assertEqual(e[1,0], 3.0)
        self.assertEqual(e[1,1], 4.0)

class PerceptronTestGroup(unittest.TestCase):
    """ Perceptron クラスのテストグループです。
    """
    def test_case_2_3_1_AND(self):
        self.assertEqual(pct.Perceptron.AND(0,0), 0)
        self.assertEqual(pct.Perceptron.AND(0,1), 0)
        self.assertEqual(pct.Perceptron.AND(1,0), 0)
        self.assertEqual(pct.Perceptron.AND(1,1), 1)

    def test_case_2_3_1_NAND(self):
        self.assertEqual(pct.Perceptron.NAND(0,0), 1)
        self.assertEqual(pct.Perceptron.NAND(0,1), 1)
        self.assertEqual(pct.Perceptron.NAND(1,0), 1)
        self.assertEqual(pct.Perceptron.NAND(1,1), 0)

    def test_case_2_3_1_OR(self):
        self.assertEqual(pct.Perceptron.OR(0,0), 0)
        self.assertEqual(pct.Perceptron.OR(0,1), 1)
        self.assertEqual(pct.Perceptron.OR(1,0), 1)
        self.assertEqual(pct.Perceptron.OR(1,1), 1)

    def test_case_2_3_1_XOR(self):
        self.assertEqual(pct.Perceptron.XOR(0,0), 0)
        self.assertEqual(pct.Perceptron.XOR(0,1), 1)
        self.assertEqual(pct.Perceptron.XOR(1,0), 1)
        self.assertEqual(pct.Perceptron.XOR(1,1), 0)

    def test_case_004(self):
        p = pct.Perceptron(0.5, 0.5)
        self.assertEqual(p.gate_and(0,0), 0)
        self.assertEqual(p.gate_and(0,1), 0)
        self.assertEqual(p.gate_and(1,0), 0)
        self.assertEqual(p.gate_and(1,1), 1)


""" This is the example code in chap 3 NeuralNetwork.
"""
class NeuralNetwork:

    @classmethod
    def step(cls, x_):
        return np.array(x_ > 0, dtype=np.int)

    @classmethod
    def sigmoid(cls, x_):
        return 1 / (1 + np.exp(-x_))

    @classmethod
    def relu(cls, x_):
        return np.maximum(0, x_)

    @classmethod
    def draw_step(cls):
        x = np.arange(-5.0, 5.0, 0.1)
        y = NeuralNetwork.step(x)
        plt.plot(x,y)
        plt.ylim(-0.1, 1.1)
        plt.show()

    @classmethod
    def draw_sigmoid(cls):
        x = np.arange(-5.0, 5.0, 0.1)
        y = NeuralNetwork.sigmoid(x)
        plt.plot(x,y)
        plt.ylim(-0.1, 1.1)
        plt.show()

    @classmethod
    def draw_relu(cls):
        x = np.arange(-5.0, 5.0, 0.1)
        y = NeuralNetwork.relu(x)
        plt.plot(x,y)
        plt.ylim(-1, 6)
        plt.show()

    @classmethod
    def softmax(cls, a_):
        c = np.max(a_)
        exp_a = np.exp(a_ - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    @classmethod
    def mean_squared_error(cls, y_, t_):
        return 0.5 * np.sum((y_ - t_)**2)


""" This is the example test code in chap 3.
"""
class NeuralNetworkTestGroup(unittest.TestCase):

    def test_array_case005(self):
        array_a = np.array([1, 2, 3, 4])
        self.assertEqual(np.ndim(array_a), 1)
        self.assertEqual(array_a.shape, (4,))
        self.assertEqual(array_a.shape[0], 4)

    def test_array_case006(self):
        array_b = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(np.ndim(array_b), 2)
        self.assertEqual(array_b.shape, (3,2))

    def test_array_case007(self):
        mat_a = np.array([[1, 2], [3, 4]])
        mat_b = np.array([[5, 6], [7, 8]])
        mat_c = np.dot(mat_a, mat_b)
        self.assertEqual(mat_a.shape, (2,2))
        self.assertEqual(mat_b.shape, (2,2))
        self.assertEqual(mat_c.shape, (2,2))

    def test_case_008_matrix(self):
        mat_x = np.array([1, 2])
        self.assertEqual(mat_x.shape, (2,))
        mat_w = np.array([[1, 3, 5], [2, 4, 6]])
        self.assertEqual(mat_w.shape, (2, 3))
        mat_y = np.dot(mat_x, mat_w)
        self.assertEqual(mat_y.shape, (3,))

    def test_case_009_neural_network(self):
        x0 = np.array([1.0, 0.5])
        w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        b1 = np.array([0.1, 0.2, 0.3])
        self.assertEqual(w1.shape, (2, 3))
        self.assertEqual(x0.shape, (2,))
        self.assertEqual(b1.shape, (3,))
        a1 = np.dot(x0, w1) + b1
        x1 = NeuralNetwork.sigmoid(a1)
        w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        b2 = np.array([0.1, 0.2])
        a2 = np.dot(x1, w2) + b2
        x2 = NeuralNetwork.sigmoid(a2)
        self.assertEqual(x2[0], 0.6262493703990729)
        self.assertEqual(x2[1], 0.7710106968556123)

    def test_case_010_softmax(self):
        a = np.array([0.3, 2.9, 4.0])
        y = NeuralNetwork.softmax(a)
        self.assertEqual(y[0], 0.01821127329554753)
        self.assertEqual(y[1], 0.24519181293507392)
        self.assertEqual(y[2], 0.7365969137693786)
        sum_y = np.sum(y)
        self.assertEqual(sum_y, 1.0)

    def test_case_011_squared(self):
        y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
        t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(np.sum(y), 1.0)
        self.assertEqual(y.shape, (10, ))
        mse = NeuralNetwork.mean_squared_error(y, t)
        self.assertEqual(mse, 0.09750000000000003)
        y2 = np.array([0.01, 0.02, 0.94, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        mse2 = NeuralNetwork.mean_squared_error(y2, t)
        self.assertEqual(mse2, 0.002300000000000003)



if __name__ == "__main__":
    try:
        unittest.main()
    except SystemExit as exp:
        pass
