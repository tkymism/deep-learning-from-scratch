import numpy as np

class Perceptron():
    """ 
    This is the example test code in chap 2.
        
    Attributes
    ----------

    Notes
    -----
    * パーセプトロンは入出力を備えたアルゴリズムである。ある入力を与えたら、決まった値が出力される。
    * パーセプトロンでは、「重み」と「バイアス」をパラメータとして設定する。
    * パーセプトロンを用いれば、AND やOR ゲートなどの論理回路を表現できる。
    * XOR ゲートは単層のパーセプトロンでは表現できない。
    * 2 層のパーセプトロンを用いれば、XOR ゲートを表現することができる。
    * 単層のパーセプトロンは線形領域だけしか表現できないのに対して、多層のパーセプトロンは非線形領域を表現することができる。
    * 多層のパーセプトロンは、（理論上）コンピュータを表現できる。    
   """

    THETA_07 = 0.7
    THETA_02 = 0.2

    # @classmethod
    # def AND(cls, x1, x2):
    #     w1, w2, theta = 0.5, 0.5, 0.7
    #     tmp = x1*w1 + x2*w2
    #     if tmp <= theta:
    #         return 0
    #     else :
    #         return 1

    @classmethod
    def AND(cls, x1, x2):
        """
        ANDのパーセプトロンになります。 

        Notes
        -----
        (0, 0) => 0
        (0, 1) => 0
        (1, 0) => 0
        (1, 1) => 1

        """
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else :
            return 1

    @classmethod
    def NAND(cls, x1, x2):
        """
        NANDのパーセプトロンになります。 

        Notes
        -----
        (0, 0) => 1
        (0, 1) => 1
        (1, 0) => 1
        (1, 1) => 0

        """
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else :
            return 1

    @classmethod
    def OR(cls, x1, x2):
        """
        ORのパーセプトロンです。 

        Notes
        -----
        (0, 0) => 0
        (0, 1) => 1
        (1, 0) => 1
        (1, 1) => 1

        """
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else :
            return 1

    @classmethod
    def XOR(cls, x1, x2):
        """ 
        XORのパーセプトロンです。 

        Notes
        -----
        (0, 0) => 0
        (0, 1) => 1
        (1, 0) => 1
        (1, 1) => 0

        """
        s1 = Perceptron.NAND(x1, x2)
        s2 = Perceptron.OR(x1, x2)
        y = Perceptron.AND(s1, s2)
        return y

    def __init__(self, w1_, w2_):
        self.w1 = w1_
        self.w2 = w2_

    def gate_and(self, x1_, x2_):
        x = np.array([x1_, x2_])
        w = np.array([self.w1, self.w2])
        tmp = np.sum(w*x) - Perceptron.THETA_07
        if tmp <= 0:
            return 0
        else:
            return 1

    def gate_nand(self, x1_, x2_):
        x = np.array([x1_, x2_])
        w = -1 * np.array([self.w1, self.w2])
        tmp = np.sum(w*x) + Perceptron.THETA_07
        if tmp <= 0:
            return 0
        else:
            return 1

    def gate_or(self, x1_, x2_):
        x = np.array([x1_, x2_])
        w = np.array([self.w1, self.w2])
        tmp = np.sum(w*x) + Perceptron.THETA_02
        if tmp <= 0:
            return 0
        else:
            return 1


