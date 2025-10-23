import pygame
import numpy as np
import sys
import random

dt= 0.1    #Krok czasowy (60 klatek na sekundę)
balls = [] # Lista piłek w symulacji
x_resolution = 1000
y_resolution = 500
gravity_vector = np.array([0, 10])
energy_conservation = 1 #Straty energii przy odbijaniu
air_resistance=0.005    #Opory powietrza

pygame.init()
window = pygame.display.set_mode((x_resolution, y_resolution))

class Ball:
    def __init__(self):
        self.pos = np.array([x_resolution / 2, y_resolution / 2]) #Położenie
        self.v = np.array([50.0, 0.0])   #Prędkość
        self.r = 10

    def simulate(self): # Główna funkcja symulująca piłkę
        self.apply_gravity()
        self.move()
        self.check_for_bounce()

    def move(self):
        self.pos += self.v*dt
        self.v *=(1-air_resistance)

    def apply_gravity(self):
        self.v += gravity_vector*dt

    def check_for_bounce(self):
        # Sprawdzamy czy piłka zderzyła się z krawędziami ekranu 
        if self.pos[0] < 0 + self.r:
            self.pos[0]= 0 + self.r
            self.v[0] *= -energy_conservation
        elif x_resolution - self.r < self.pos[0]:
            self.v[0] *= -energy_conservation
            self.pos[0]= x_resolution - self.r

        if self.pos[1] < 0 + self.r:
            self.pos[1]= 0 + self.r
            self.v[1] *= -energy_conservation
        elif y_resolution - self.r < self.pos[1]:
            self.v[1] *= -energy_conservation
            self.pos[1]= y_resolution - self.r

def collision(b1, b2):
    delta_pos = b2.pos - b1.pos #wektor między środkami piłek
    dist = np.linalg.norm(delta_pos)    #odległość między środkami piłek
    if dist > b1.r + b2.r: #brak kolizji
        return  

    n = delta_pos / dist    #wektor normalny 
    t = np.array([-n[1], n[0]]) #wektor styczny

    #Rzutowanie prędkości piłek na wektory n i t
    v1n = np.dot(b1.v, n)
    v1t = np.dot(b1.v, t)
    v2n = np.dot(b2.v, n)
    v2t = np.dot(b2.v, t)

    #Wymiana pędu, zakładam równe masy piłek
    b1.v = v2n * n + v1t * t
    b2.v = v1n * n + v2t * t

    #Zmiana położenia (by piłki nie przenikały się)
    correction = (b1.r + b2.r - dist) / 2 * n
    b1.pos -= correction
    b2.pos += correction

# Tworzę piłki
for i in range(11):
    balls.append(Ball())
    balls[i].pos[0] = random.randint(1, x_resolution)
    balls[i].pos[1] = random.randint(1, y_resolution)

# Zaczynam symulację
run_simulation = True
while run_simulation:
    window.fill((255, 255, 255))    #Okno koloru białego
    for ball in balls:
        ball.simulate()

        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                collision(balls[i], balls[j])

        pygame.draw.circle(window, (0, 0, 255), [int(ball.pos[0]), int(ball.pos[1])] , ball.r)

    pygame.display.flip()   #Odświeżenie ekranu
    pygame.time.Clock().tick(60)    #Ograniczenie liczby klatek na sekundę

    for e in pygame.event.get():

        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif e.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = np.array(pygame.mouse.get_pos())
            for ball in balls:
                if np.linalg.norm(ball.pos - mouse_pos) <= ball.r:
                    direction = ball.pos - mouse_pos    #wektor do środka piłki od miejsca kliknięcia
                    if np.linalg.norm(direction) != 0:
                        direction = direction / np.linalg.norm(direction)   #normalizacja wektora
                    ball.v = direction * 200  # nadaje prędkość po kliknięciu





    
