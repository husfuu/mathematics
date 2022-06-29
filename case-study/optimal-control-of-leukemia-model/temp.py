    lambda_S[N-1], lambda_I[N-1], lambda_W[N-1] = 0, 0, 0
    u_1[0] = 0; u_2[0] = 0;

    for i in reversed(range(N+1)):
        k_1 = dHdS(lambda_S[i], lambda_I[i], lambda_W[i], I,  a_0, beta3, u_1_prev[i])
        l_1 = dHdI(lambda_S[i], lambda_I[i], lambda_W[i], S[i], b_0, beta3, gamma, u_2_prev[i])
        m_1 = dHdW(lambda_W[i], c_0)

        k_2 = dHdS(lambda_S[i] - (k_1/2), lambda_I[i] - (k_1/2), I, a_0, beta3, u_1_prev[i])
        l_2 = dHdI(lambda_S[i] - (k_1/2), lambda_I[i] - (k_1/2), S[i], b_0, beta3, gamma, u_2_prev[i])
        m_2 = dHdW(lambda_W[i], c_0)

        k_3 = dHdS(lambda_S[i] - (k_2/2), lambda_I[i] - (k_2/2), I, a_0, beta3, u_1_prev[i])
        l_3 = dHdI(lambda_S[i] - (k_2/2), lambda_I[i] - (k_2/2), S[i], b_0, beta3, gamma, u_2_prev[i])
        m_3 = dHdW(lambda_W[i] - (l_2/2), c_0)

        k_4 = dHdS(lambda_S[i] - (k_3), lambda_I[i] - (k_3), I, a_0, beta3, u_1_prev[i])
        l_4 = dHdI(lambda_S[i] - (k_3), lambda_I[i] - (k_3), S[i], b_0, beta3, gamma, u_2_prev[i])
        m_4 = dHdW(lambda_W[i] - (l_3), c_0)

        # update u_1 and u_2
        u_1 = update_u_1(u_1[i], lambda_I[i], lambda_W[i], I[i], M_1)
        u_2 = update_u_2(u_2[i], lambda_I[i], lambda_W[i], I[i], M_2)

        # update lambdas
        lambda_S[i+1] = lambda_S[i] - 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        lambda_I[i+1] = lambda_I[i] - 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        lambda_W[i+1] = lambda_W[i] - 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
    
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