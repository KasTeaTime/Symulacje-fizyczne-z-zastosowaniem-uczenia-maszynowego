import pygame
import numpy as np
import sys
import random
#Wykorzystuje wzór 10 artykułu

dt= 0.05    #Krok czasowy 
x_resolution = 1000
y_resolution = 500
gravity_vector = np.array([0, 100])
num_beads = 6
beads = []
push_strength = 4e4  #siła „pchnięcia” koralika po kliknięciu
gravity_enabled = False  #stan początkowy grawitacji

pygame.init()
window = pygame.display.set_mode((x_resolution, y_resolution))

#Parametry okręgu na którym będą koralki (drutu)
wire_center = np.array([x_resolution / 2, y_resolution / 2], dtype=float)
wire_radius = min(x_resolution, y_resolution) * 0.4

class Bead:
    def __init__(self, pos, r, m=None):
        self.pos = np.array(pos, dtype=float)
        self.prev_pos = np.copy(self.pos)
        self.v = np.zeros(2)
        self.r = r
        self.m = m if m is not None else np.pi * r * r #Masa koralika na podstawie promienia

    def apply_gravity(self, gravity_on):
        if gravity_on and self.m < 1e5:  #brak grawitacji dla centralnej kulki jeśli damy masę powyżej 1e5
            self.v += gravity_vector * dt

    def move(self):
        self.prev_pos[:] = self.pos #Aktualizacja zawartości tablicy (kopia w miejsce poprzednich wartości)
        self.pos += self.v*dt

    def update_velocity(self):
        self.v = (self.pos - self.prev_pos) / dt

    #Korekcja położenia koralików
    def apply_constraint_with_center(self, center_bead, wire_radius):
        delta = self.pos - center_bead.pos  #Wektor od centrum drutu do koralika
        dist = np.linalg.norm(delta)
        n = delta / dist

        w1 = 1.0 / self.m
        w2 = 1.0 / center_bead.m
        correction = (dist - wire_radius) / (w1 + w2)

        self.pos -= w1 * correction * n  #Przesunięcie koralika
        center_bead.pos += w2 * correction * n  #Niemal nieruchomy, bo masa duża

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

#Tworzenie centralnej kulki (drutu)
center_bead = Bead(wire_center, r=5, m=1e5)  # ogromna masa => nieruchoma

#Tworzenie koralików
angle = 0.0
for _ in range(num_beads):
    r = random.uniform(10, 25)  #Losuje promień koralika z przedziału 10-25
    pos = wire_center + wire_radius * np.array([np.cos(angle), np.sin(angle)])
    beads.append(Bead(pos, r))
    angle += 2 * np.pi / num_beads  #Zwiększenie kąta dla następnego koralika

# Zaczynam symulację
run_simulation = True
clock = pygame.time.Clock()

while run_simulation:
    window.fill((255, 255, 255))

    #Symulacja koralików
    for bead in beads:
        bead.apply_gravity(gravity_enabled)
        bead.move()
        bead.apply_constraint_with_center(center_bead, wire_radius)
        bead.update_velocity()

    #Kolizje koralików
    for i in range(len(beads)):
        for j in range(i + 1, len(beads)):
            collision(beads[i], beads[j])

    #Rysowanie
    pygame.draw.circle(window, (255, 0, 0), center_bead.pos.astype(int), int(wire_radius), width=2)
    for bead in beads:
        pygame.draw.circle(window, (0, 0, 255), bead.pos.astype(int), int(bead.r))

    #Tekst o stanie grawitacji
    font = pygame.font.SysFont(None, 24)
    label = "Grawitacja: WŁĄCZONA" if gravity_enabled else "Grawitacja: WYŁĄCZONA"
    text = font.render(label, True, (0, 0, 0))
    window.blit(text, (20, 20))

    pygame.display.flip()
    clock.tick(60)

    #Obsługa zdarzeń
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif e.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
            for bead in beads:
                if np.linalg.norm(bead.pos - mouse_pos) <= bead.r:
                    direction = bead.pos - mouse_pos    #wektor do środka koralika od miejsca kliknięcia
                    dist = np.linalg.norm(direction)
                    if dist > 0:
                        direction /= dist
                    bead.v += direction * push_strength / bead.m

        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_g:
                gravity_enabled = not gravity_enabled  #przełączanie grawitacji
