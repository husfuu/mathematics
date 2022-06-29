import numpy as np
from utility import dSdt, dIdt, dWdt, dHdS, dHdI, dHdW, update_u_1, update_u_2

# paramater values
h = 0.5; t = 200; N = int(t/h);

# initial values in 3 compartments
S0 = 150; I0 = 9; W0 = 2;

# params
A = 1.5
a_0 = 0.01
b_0 = 0.003
c_0 = 0.03

beta1 = 0.0002
beta2 = 0.0003
beta3 = 0.0004
gamma = 0.0001

tes = 1

S = np.zeros(N); S_prev = np.zeros(N)
I = np.zeros(N); I_prev = np.zeros(N)
W = np.zeros(N); W_prev = np.zeros(N)

# assign initial value to state variable
S[0], I[0], W[0] = S0, I0, W0;

lambda_S = np.zeros(N); lambda_S_prev = np.zeros(N)
lambda_I = np.zeros(N); lambda_I_prev = np.zeros(N)
lambda_W = np.zeros(N); lambda_W_prev = np.zeros(N)

lambda_S[N-1], lambda_I[N-1], lambda_W[N-1] = 0, 0, 0

u_1 = np.zeros(N); u_1_prev = np.zeros(N)
u_2 = np.zeros(N); u_2_prev = np.zeros(N)

M_1 = 0.1
M_2 = 0.1

while tes > 1e-3:
    S_prev, I_prev, W_prev = S, I, W;
    lambda_S_prev, lambda_I_prev, lambda_W_prev = lambda_S, lambda_I, lambda_W;
    u_1_prev, u_2_prev = u_1, u_2;

    # assign initial value to state variable
    S[0], I[0], W[0] = S0, I0, W0;

    # forward
    for i in range(N-1):
        k_1 = dSdt(S[i], I[i], A, a_0, beta3, u_1_prev[i])
        l_1 = dIdt(S[i], I[i], beta3, gamma, b_0, u_2_prev[i])
        m_1 = dWdt(S[i], I[i], W[i], gamma, c_0, u_1[i], u_2_prev[i])

        k_2 = dSdt(S[i] + k_1*(h/2), I[i] + l_1*(h/2), A, a_0, beta3, u_1_prev[i])
        l_2 = dIdt(S[i] + k_1*(h/2), I[i] + l_1*(h/2), beta3, gamma, b_0, u_2_prev[i])
        m_2 = dWdt(S[i] + k_1*(h/2), I[i] + l_1*(h/2), W[i] + m_1*(h/2), gamma, c_0, u_1_prev[i], u_2_prev[i])

        k_3 = dSdt(S[i] + k_2*(h/2), I[i] + l_2*(h/2), A, a_0, beta3, u_1_prev[i])
        l_3 = dIdt(S[i] + k_2*(h/2), I[i] + l_2*(h/2), beta3, gamma, b_0, u_2_prev[i])
        m_3 = dWdt(S[i] + k_2*(h/2), I[i] + l_2*(h/2), W[i] + m_2*(h/2), gamma, c_0, u_1_prev[i], u_2_prev[i])

        k_4 = dSdt(S[i] + k_3*h, I[i] + l_3*h, A, a_0, beta3, u_1_prev[i])
        l_4 = dIdt(S[i] + k_3*h, I[i] + l_3*h, beta3, gamma, b_0, u_2_prev[i])
        m_4 = dWdt(S[i] + k_3*h, I[i] + l_3*h, W[i] + m_3*h, gamma, c_0, u_1_prev[i], u_2_prev[i])

        # update value of state variable
        S[i+1] = S[i] + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)*h 
        I[i+1] = I[i] + 1/6 * (l_1 + 2*l_2 + 2*l_3 + l_4)*h
        W[i+1] = W[i] + 1/6 * (m_1 + 2*m_2 + 2*m_3 + m_4)*h
    
    lambda_S[N-1], lambda_I[N-1], lambda_W[N-1] = 0, 0, 0
    u_1[0] = 0; u_2[0] = 0;

    for i in reversed(range(N)):
        k_1 = dHdS(lambda_S[i], lambda_I[i], lambda_W[i], I[i], a_0, beta3, u_1_prev[i])
        l_1 = dHdI(lambda_S[i], lambda_I[i], lambda_W[i], S[i], b_0, beta3, gamma, u_2_prev[i])
        m_1 = dHdW(lambda_W[i], c_0)
        
        # print(k_1, l_1, m_1)

        k_2 = dHdS(lambda_S[i] - (k_1/2), lambda_I[i] - (k_1/2), lambda_W[i] - (k_1/2), I[i], a_0, beta3, u_1_prev[i])
        l_2 = dHdI(lambda_S[i] - (k_1/2), lambda_I[i] - (k_1/2), lambda_W[i] - (k_1/2), S[i], b_0, beta3, gamma, u_2_prev[i])
        m_2 = dHdW(lambda_W[i] - (l_1/2), c_0)

        k_3 = dHdS(lambda_S[i] - (k_2/2), lambda_I[i] - (k_2/2), lambda_W[i] - (k_2/2), I[i], a_0, beta3, u_1_prev[i])
        l_3 = dHdI(lambda_S[i] - (k_2/2), lambda_I[i] - (k_2/2), lambda_W[i] - (k_2/2), S[i], b_0, beta3, gamma, u_2_prev[i])
        m_3 = dHdW(lambda_W[i] - (l_2/2), c_0)

        k_4 = dHdS(lambda_S[i] - (k_3), lambda_I[i] - (k_3), lambda_W[i] - (k_3), I[i], a_0, beta3, u_1_prev[i])
        l_4 = dHdI(lambda_S[i] - (k_3), lambda_I[i] - (k_3), lambda_W[i] - (k_3), S[i], b_0, beta3, gamma, u_2_prev[i])
        m_4 = dHdW(lambda_W[i] - (l_3), c_0)

        # update u_1 and u_2
        u_1[i-1] = update_u_1(u_1[i], lambda_I[i], lambda_W[i], I[i], M_1)
        u_2[i-1] = update_u_2(u_2[i], lambda_I[i], lambda_W[i], I[i], M_2)

        # update lambdas
        lambda_S[i] = lambda_S[i] - 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        lambda_I[i] = lambda_I[i] - 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        lambda_W[i] = lambda_W[i] - 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)

    errS = sum(abs(S - S_prev))
    errI = sum(abs(I - I_prev))
    errW = sum(abs(W - W_prev))

    errLambda_S = sum(abs(lambda_S - lambda_S_prev))
    errLambda_I = sum(abs(lambda_I - lambda_I_prev))
    errLambda_W = sum(abs(lambda_W - lambda_W_prev))

    errU_1 = sum(abs(u_1 - u_1_prev))
    errU_2 = sum(abs(u_2 - u_2_prev))

    tes = errS + errI + errW + errLambda_S + errLambda_I + errLambda_W + errU_1 + errU_2
    u_1 = (0.5*u_1 + 0.5*u_1_prev)
    u_2 = (0.5*u_2 + 0.5*u_2_prev)
    print(tes)


# print(lambda_S)
# print("-----------------")
# print(lambda_I)
# print("-----------------")
# print(lambda_W)
# print("-----------------")

# print(lambda_S)
# print("-----------------")
# print(lambda_I)
# print("-----------------")
# print(lambda_W)
# print("-----------------")