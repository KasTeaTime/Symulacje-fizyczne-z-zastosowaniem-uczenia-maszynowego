import pygame
import numpy as np
import sys
import random

dt= 0.05    #Krok czasowy 
x_resolution = 1000
y_resolution = 500
gravity_vector = np.array([0, 100])
num_beads = 6
beads = []

pygame.init()
window = pygame.display.set_mode((x_resolution, y_resolution))

#Parametry okręgu na którym będą koralki (drutu)
wire_center = np.array([x_resolution / 2, y_resolution / 2], dtype=float)
wire_radius = min(x_resolution, y_resolution) * 0.4

prev_wire_center = np.copy(wire_center) #Kopia położenia centrum okręgu
wire_speed = 30.0  # prędkość ruchu okręgu [px/s]

class Bead:
    def __init__(self, pos, r):
        self.pos = np.array(pos, dtype=float)
        self.prev_pos = np.copy(self.pos)
        self.v = np.zeros(2)
        self.r = r
        self.m = np.pi * r * r     #Masa koralika na podstawie promienia

    def simulate(self, wire_center, wire_velocity):
        self.apply_gravity()
        self.move()
        self.keep_on_wire(wire_center)
        self.update_velocity(wire_velocity)

    def apply_gravity(self):
        self.v += gravity_vector * dt

    def move(self):
        self.prev_pos[:] = self.pos #Aktualizacja zawartości tablicy (kopia w miejsce poprzednich wartości)
        self.pos += self.v*dt

    def keep_on_wire(self, wire_center):
        dir_vec = self.pos - wire_center    #wektor od środka okręgu do koralika
        length = np.linalg.norm(dir_vec)    #odległość koralika od środka
        dir_vec /= length   #normalizacja
        correction = wire_radius - length   #
        self.pos += dir_vec * correction    #przesunięcie koralika na okrąg

    def update_velocity(self, wire_velocity):   #Aktualizacja prędkości koralika uwzględniając ruch poręczy
        self.v = (self.pos - self.prev_pos) / dt + wire_velocity    #prędkość = prędkość poruszającego się koralika + prękość obręczy

def collision(b1, b2):
    delta_pos = b2.pos - b1.pos #wektor między środkami piłek
    dist = np.linalg.norm(delta_pos)    #odległość między środkami piłek
    if dist == 0 or dist > b1.r + b2.r: #brak kolizji
        return

    n = delta_pos / dist    #wektor normalny
    t = np.array([-n[1], n[0]]) #wektor styczny

    #Rzutowanie prędkości piłek na wektory n i t
    v1n = np.dot(b1.v, n)
    v1t = np.dot(b1.v, t)
    v2n = np.dot(b2.v, n)
    v2t = np.dot(b2.v, t)

    #Składowe normalne prędkości po sprężystym zderzeniu koralików o różnych masach
    m1 = b1.m
    m2 = b2.m
    new_v1n = (m1 * v1n + m2 * v2n - m2 * (v1n - v2n)) / (m1 + m2)
    new_v2n = (m1 * v1n + m2 * v2n - m1 * (v2n - v1n)) / (m1 + m2)

    #Wymiana pędu
    b1.v = new_v1n * n + v1t * t
    b2.v = new_v2n * n + v2t * t

    #Zmiana położenia (by piłki nie przenikały się)
    correction = (b1.r + b2.r - dist) / 2 * n
    b1.pos -= correction
    b2.pos += correction

# --- Tworzenie koralików -------------------------------------------------
angle = 0.0
for _ in range(num_beads):
    r = random.uniform(10, 25)  #Losuje promień koralika z przedziału 10-25
    pos = wire_center + wire_radius * np.array([np.cos(angle), np.sin(angle)]) #Pozycja koralika na okręgu na podstawie kąta
    beads.append(Bead(pos, r))  
    angle += np.pi / num_beads #Zwiększenie kąta dla następnego koralika 

# Zaczynam symulację
run_simulation = True
while run_simulation:
    window.fill((255, 255, 255))    #Okno koloru białego
 
    prev_wire_center[:] = wire_center   #Kopia położenia okręgu

    #Sterowanie okręgiem klawiaturą
    keys = pygame.key.get_pressed()
    move = np.zeros(2, dtype=float)
    if keys[pygame.K_LEFT]:
        move[0] -= wire_speed * dt
    if keys[pygame.K_RIGHT]:
        move[0] += wire_speed * dt
    if keys[pygame.K_UP]:
        move[1] -= wire_speed * dt
    if keys[pygame.K_DOWN]:
        move[1] += wire_speed * dt
    wire_center += move

    wire_velocity = (wire_center - prev_wire_center) / dt   #Prędkość obręczy

    pygame.draw.circle(window, (255, 0, 0), wire_center.astype(int), int(wire_radius), width=2)

    for bead in beads:
        bead.simulate(wire_center, wire_velocity)

    for i in range(len(beads)):
        for j in range(i + 1, len(beads)):
            collision(beads[i], beads[j])

    for bead in beads:
        pygame.draw.circle(window, (0, 0, 255), bead.pos.astype(int), int(bead.r))

    pygame.display.flip()   #Odświeżenie ekranu
    pygame.time.Clock().tick(60)    #Ograniczenie liczby klatek na sekundę
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
