import numpy as np
import sys


def build_weight_matrix(p):
    """根据伪逆规则生成权值矩阵"""
    p = np.array(p)
    p_ = np.linalg.inv(p.T.dot(p)).dot(p.T)
    wm = p.dot(p_)  # 自联想记忆模型的目标输出向量等同于输入向量
    return wm


def identify(p, input_vector):
    """对输入向量进行模式识别"""
    wm = build_weight_matrix(p)
    result = wm.dot(input_vector)
    # 下面的处理起到硬限值传输函数的作用
    for i in range(len(result)):
        result[i] = 1 if result[i] > 0 else -1
    return result


def print_letter(letter):
    """打印11*6点阵表示的字母"""
    bit_list = [letter[:11], letter[11:22], letter[22:33], letter[33:44], letter[44:55], letter[55:]]
    bit_list = np.array(bit_list).T
    new_list = []
    for row in bit_list:
        new_list.append(list(row))
    for i in range(len(new_list)):
        for j in range(len(new_list[i])):
            if new_list[i][j] == 1:
                new_list[i][j] = '*'
            else:
                new_list[i][j] = ' '
            print(new_list[i][j], end='')
        print()


if __name__ == '__main__':
    # 下面输入的"SCAU"的11*6点阵表示(逐列扫描)
    S = [-1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1,
         1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1,
         1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1,
         1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1,
         1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1,
         -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1]
    C = [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
         -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
         1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
         -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    A = [-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1,
         -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1,
         1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1,
         1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1,
         -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
    U = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1]

    p = np.array([S, C, A, U]).T


    def show_result(input_letter):
        print_letter(input_letter)
        result = identify(p, np.array(input_letter))
        print(result)
        print(result == input_letter)
        print()


    show_result(S)
    show_result(C)
    show_result(A)
    show_result(U)

print(sys.version)


