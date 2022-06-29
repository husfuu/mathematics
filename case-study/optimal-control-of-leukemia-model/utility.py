# STATE
def dSdt(S, I, A, a_0, beta, u_1):
    return (A - a_0*S - beta*S*I - u_1*S)

def dIdt(S, I, beta, gamma, b_0, u_2):
    return (beta*S*I - (b_0 + gamma)*I - u_2*I)

def dWdt(S, I, W, gamma, c_0, u_1, u_2):
    return (gamma*I - c_0*W + u_1*S + u_2 *I)

# COSTATE
def dHdS(lambda_S, lambda_I, lambda_W, I, a_0, beta, u_1):
    return -1*((-a_0*lambda_S) - (beta*I*lambda_S) - (u_1*lambda_S) + (beta*I*lambda_I) + (u_1*lambda_W))

def dHdI(lambda_S, lambda_I, lambda_W, S, b_0, beta, gamma, u_2):
    return (-1*(1 - (beta*S*lambda_S) + (beta*S*lambda_I) - ((b_0 + gamma)*lambda_I) - (u_2*lambda_I) + (gamma*lambda_W) + u_2*lambda_W))

def dHdW(lambda_W, c_0):
    return (-1*(-c_0*lambda_W))

# update u_1 and u_2
def update_u_1(u_1, lambda_I, lambda_W, I, M_1):
    if (u_1 <= 0):
        return 0
    elif ((u_1 > 0) or (u_1 < 1)):
        print("disini salah")
        return ((lambda_I - lambda_W)*I / M_1)
    else:
        return 1

def update_u_2(u_2, lambda_I, lambda_W, I, M_2):
    if (u_2 <= 0):
        return 0
    elif ((0 < u_2) or (u_2 < 1)):
        return ((lambda_I - lambda_W)*I / M_2)
    else:
        return 1