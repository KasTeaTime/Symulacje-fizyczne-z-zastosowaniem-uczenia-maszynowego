import sympy
import numpy as np
from scipy.integrate import solve_ivp
from sympy import lambdify, diff

#Parametry metryki Schwarzschilda
masa=10
C=1
G=1
T = 70

#Ruch masowej cząstki próbnej 
initial_value = [0, 30, np.pi/2, 0, 1, 0, 0, 0]

A=2*G*masa/C/C #Parametr pomocniczy
print(A, "promień Schwarzschilda")

def macierz4x4(wartosc):
    return [[wartosc for i in range(4)] for j in range(4)]

def metrykaSchwarzschilda(): #(d tau)^2
    M = macierz4x4(0)
    M[0][0]=(1-A/wspolrzedne[1])
    M[1][1]=-1/(1-A/wspolrzedne[1])/C**2
    M[2][2]=-wspolrzedne[1]*wspolrzedne[1]/C**2
    M[3][3]=-wspolrzedne[1]*wspolrzedne[1]*(sympy.sin(wspolrzedne[2]) ** 2)/C**2
    return M

def metrykaMinkowskiego(): #(d tau)^2
    M = macierz4x4(0)
    M[0][0]=1
    M[1][1]=-1
    M[2][2]=-1
    M[3][3]=-1
    return M

def macierzOdwrotna(macierz):  #dla macierzy diagonalnej
    m = macierz4x4(0)
    for i in range(4):
        m[i][i] = 1/macierz[i][i]
    return m

def OblSymChristoffela(metryka,metrykaOdwrotna,wspolrzedne): 
    ch=[macierz4x4(0) for i in range(4)]
    for a in range(len(wspolrzedne)):
        for i in range(len(wspolrzedne)):    
            for j in range(len(wspolrzedne)):
                for k in range(len(wspolrzedne)):
                    ch[a][i][j]=ch[a][i][j]+0.5*(metrykaOdwrotna[a][k])*((diff(metryka[j][k],wspolrzedne[i]))+(diff(metryka[i][k],wspolrzedne[j]))-(diff(metryka[i][j],wspolrzedne[k])))
    return ch                


def F(t,tab):   #Funkcja do rozwiązania równania różniczkowego 
    u = np.array(tab[0:4])
    v = np.array(tab[4:8])

    chris = func(u[0],u[1], u[2],u[3])

    du = v
    dv = -np.dot(np.dot(chris, v), v)

    return np.concatenate((du, dv))


wspolrzedne = sympy.symbols('t r theta phi')     #utworzenie tablicy symboli numerycznych

#M=metrykaMinkowskiego()
M=metrykaSchwarzschilda()

N=macierzOdwrotna(M)

SymboleChristofela=OblSymChristoffela(M,N,wspolrzedne)

func = sympy.lambdify((wspolrzedne[0], wspolrzedne[1], wspolrzedne[2],wspolrzedne[3]), SymboleChristofela,("numpy"))

t_eval = np.linspace(0, T, int(T * 123 + 1))

# rozwiązanie równania geodezyjnych
sol = solve_ivp(F, [0, T], initial_value, t_eval=t_eval)


import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6),)
plt.plot(sol.y[0], sol.y[1])
ax = plt.gca()
plt.grid()
plt.xlabel('t')
plt.ylabel('r')

plt.show()
