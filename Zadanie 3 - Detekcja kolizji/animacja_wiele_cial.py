#Sterowanie: 1 - Bounding Volume Hierarchy, 2 - Sweep and Prune, 3 - Brute Force
import sys
import time
import random
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pymorton import interleave3

# Parametry konfiguracji
box_count = 1000
world_size = 100.0
min_box_size = 1.0
max_box_size = 10.0
max_speed = 0.7
window_size = (1280, 720)

#Domyślny algorytm: 'bvh', 'sweep', 'bruteforce'
algorithm = 'bvh'

class Box:
    def __init__(self, idx):
        self.id = idx
        self.width = random.randint(min_box_size, max_box_size)
        self.height = random.randint(min_box_size, max_box_size)
        self.depth = random.randint(min_box_size, max_box_size)

        self.pos = np.array([
            random.randint(-world_size/2, world_size/2),
            random.randint(-world_size/2, world_size/2),
            random.randint(-world_size/2, world_size/2)
        ], dtype=np.float32)

        self.vel = np.array([   #od -0.5 max_speed do 0.5 max_speed
            (random.random() - 0.5) * max_speed,
            (random.random() - 0.5) * max_speed,
            (random.random() - 0.5) * max_speed
        ], dtype=np.float32)

        self.color = (
    random.uniform(0.3, 1.0),
    random.uniform(0.3, 1.0),
    random.uniform(0.3, 1.0))
        self.is_colliding = False

    def update(self):
        self.pos += self.vel    #Brak reakcji na kolizję
        half = world_size / 2.0 
        for i in range(3):  #Kołowe warunki brzegowe
            if self.pos[i] > half:
                self.pos[i] = -half
            elif self.pos[i] < -half:
                self.pos[i] = half
        self.is_colliding = False   #Czyszczenie flag kolizji

    def get_aabb(self): #granice pudełka
        half_sizes = np.array([self.width / 2.0, self.height / 2.0, self.depth / 2.0])
        return self.pos - half_sizes, self.pos + half_sizes

class BVHNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.box_id = -1
        self.a_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        self.a_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    def is_leaf(self):  #Węzeł czy liść
        return self.box_id != -1    

#Sprawdzenie przecięcia prostopadłościanów (AABB– Axis-Aligned Bounding Boxes)
def aabb_intersect(a_min, a_max, b_min, b_max):
    return np.all(a_min <= b_max) and np.all(a_max >= b_min)    #True/False- do kolizji potrzebujemy pokrycia dla 3 osi

def calculate_morton_code(x, y, z):

    def norm(v):    #Do przedzaiłu [0,1]
        return (v + world_size / 2.0) / world_size
    
    x = min(max(norm(x), 0.0), 1.0)
    y = min(max(norm(y), 0.0), 1.0)
    z = min(max(norm(z), 0.0), 1.0)
    xi = min(int(x * 1023), 1023)   #Współrzędne do całkowitej 10-bitowej liczby
    yi = min(int(y * 1023), 1023)
    zi = min(int(z * 1023), 1023)
    return interleave3(xi, yi, zi)  #Bity w kod Mortona

#Konstrukcja BVH (liść drzewa to 1 pudełko)
def create_leaf(box_id, boxes):
    node = BVHNode()
    node.box_id = box_id
    a_min, a_max = boxes[box_id].get_aabb()
    node.a_min = a_min.copy()
    node.a_max = a_max.copy()
    return node

def create_subtree(lst, begin, end, boxes):
    if begin == end:    #lst zawiera jeden element
        return create_leaf(lst[begin]['id'], boxes)
    
    #Rekurencyjne budowanie poddrzewa 
    mid = (begin + end) // 2
    node = BVHNode()
    node.left = create_subtree(lst, begin, mid, boxes)
    node.right = create_subtree(lst, mid + 1, end, boxes)
    node.a_min = np.minimum(node.left.a_min, node.right.a_min)  #Obwiednia węzła
    node.a_max = np.maximum(node.left.a_max, node.right.a_max)
    return node

def create_bvh(boxes):
    lst = []
    for i, b in enumerate(boxes):
        code = calculate_morton_code(b.pos[0], b.pos[1], b.pos[2])
        lst.append({'id': i, 'morton': code})
    lst.sort(key=lambda x: x['morton'])
    return create_subtree(lst, 0, len(lst) - 1, boxes)

#Detekcja kolizji Bounding Volume Hierarchy
def find_collisions_bvh(box_id, box, node, boxes, collisions, check_count):
    check_count[0] += 1

    a_min, a_max = box.get_aabb()
    if not aabb_intersect(a_min, a_max, node.a_min, node.a_max): #Czy pudełko w węźle
        return
    
    if node.is_leaf():
        if node.box_id != box_id:
            other = boxes[node.box_id]
            bmin, bmax = other.get_aabb()
            if aabb_intersect(a_min, a_max, bmin, bmax):
                collisions.append((box_id, node.box_id))
        return
    find_collisions_bvh(box_id, box, node.left, boxes, collisions, check_count)
    find_collisions_bvh(box_id, box, node.right, boxes, collisions, check_count)

def check_collisions_bvh(boxes, root):
    collisions = []
    check_count = [0]   #Zmienna do zliczania kolizji, jest w fotmie tab bo potrzebuję przekazywana przez referencje
    for i, b in enumerate(boxes):
        find_collisions_bvh(i, b, root, boxes, collisions, check_count)
    return collisions, check_count[0]

#Detekcja kolizji Brute-force
def check_collisions_bruteforce(boxes):
    collisions = []
    check_count = int(box_count*(box_count-1)/2)

    n = len(boxes)
    for i in range(n):
        a_min, a_max = boxes[i].get_aabb()
        for j in range(i + 1, n):
            b_min, b_max = boxes[j].get_aabb()
            if aabb_intersect(a_min, a_max, b_min, b_max):
                collisions.append((i, j))
    return collisions, check_count

#Detekcja kolizji Sweep and Prune (sortowanie w osi x a następnie testy)
def check_collisions_sweep_and_prune(boxes):
    intervals = []  #Lista krotek (minx, maxx, id)
    active = []  #Lista krotek (maxx, id)
    collisions = []
    check_count = 0

    for b in boxes:
        a_min, a_max = b.get_aabb()
        intervals.append((a_min[0], a_max[0], b.id))
    intervals.sort(key=lambda it: it[0])    #Po minx

    for minx, maxx, b in intervals:
        #Usuwanie pudełek które kończą się przed bieżącym minx
        new_active = []
        for i in active:
            if i[0] >= minx:
                new_active.append(i)
        active = new_active
        
        #Porównanie z pozostałymi pudełkami
        for maxx_act, act in active:
            check_count += 1
            a_min, a_max = boxes[act].get_aabb()
            b_min, b_max = boxes[b].get_aabb()
            if aabb_intersect(a_min, a_max, b_min, b_max):
                collisions.append((act, b))
        active.append((maxx, b))    #Pudełko z tej pętli może kolidować z innymi (bo spr tylko do mniejszego minx tzn tych z stworzonego wcześneij active)
    return collisions, check_count

# Rysowanie kostki
def draw_cube(half_sizes):      #rysowanie sześcianu wyśrodkowanego w punkcie początkowym i skalowanego do rozmiarów połówkowych
    hx, hy, hz = half_sizes

    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),    #Wierzchołki sześcianu
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    
    faces = [(0, 1, 2, 3), (4, 5, 6, 7),    #Ściany
        (0, 1, 5, 4), (2, 3, 7, 6),
        (1, 2, 6, 5), (0, 3, 7, 4)]
    
    normals = [(0, 0, -1), (0, 0, 1),   #Wektory normalne ścian
        (0, -1, 0), (0, 1, 0),
        (1, 0, 0), (-1, 0, 0)]
    
    glBegin(GL_QUADS)
    for fi, face in enumerate(faces):
        glNormal3fv(normals[fi])
        for idx in face:
            glVertex3fv(v[idx])
    glEnd()

def init_opengl():  #Inicjalizacja
    glEnable(GL_DEPTH_TEST)     #Test głębokości, aby obiekty bliższe przesłaniały dalsze
    glDepthFunc(GL_LESS)        #Rysowanie gdy piksel bliżej kamery
    glCullFace(GL_BACK)         #Nie rysuje ściań z tyłu obiektów (niewidocznych)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 1.0, 1.0, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

#Tworzy macierz projekcji (do przekształcenia współrzędnych by obiekty dalsze wydawały się mniejsze)
def perspective(fov, screen, near, far):    #screen = szerokość / wysokość ekanu
    f = 1.0 / np.tan(np.radians(fov) / 2)  #Ogniskowa obrazu
    M = np.array([
        [f / screen, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)
    glMultMatrixf(M.T)

def main():
    global algorithm    #Czyta zmienną algorithm
    pygame.init()
    pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)    #DOUBLEBUF by obiekty były wyświetlane na raz

    #Modyfikacja macierzy projekcji 
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()    #Reset do macierzy jednostkowej 
    perspective(45, window_size[0] / window_size[1], 0.1, 1000.0) #Projekcja * persektywa

    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -150.0)  #Ustawienia kamery tak by widać było całą planszę
    init_opengl()

    boxes = [Box(i) for i in range(box_count)] #Tworzenie pudełek
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_1:
                    algorithm = 'bvh'
                elif event.key == K_2:
                    algorithm = 'sweep'
                elif event.key == K_3:
                    algorithm = 'bruteforce'

        for b in boxes:
            b.update()

        t1 = time.time()
        if algorithm == 'bvh':
            root = create_bvh(boxes)
            collisions, checks = check_collisions_bvh(boxes, root)
        elif algorithm == 'sweep':
            collisions, checks = check_collisions_sweep_and_prune(boxes)
        else:
            collisions, checks = check_collisions_bruteforce(boxes)
        t2 = time.time()

        # Oznaczenie kolidujących pudełek
        for a, b in collisions:
            boxes[a].is_colliding = True
            boxes[b].is_colliding = True

        # Wyświetlanie
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()      #Zapis aktualnej macierzy transformacji

        # Rysowanie planszy (sześcianu)
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.2, 0.2)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glPushMatrix()
        glScalef(world_size, world_size, world_size)
        draw_cube((0.5, 0.5, 0.5))
        glPopMatrix()       #Przywrócenie ostatnio zapisanej macierzy transformacji
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)

        # Rysowanie pudełek
        for b in boxes:
            glPushMatrix()
            glTranslatef(b.pos[0], b.pos[1], b.pos[2])
            half = (b.width / 2.0, b.height / 2.0, b.depth / 2.0)
            if b.is_colliding:
                glColor3f(1.0, 0.2, 0.0)    #Czerwony
                
            else:
                glColor3fv(b.color)

            draw_cube(half)
            glPopMatrix()

        glPopMatrix()
        pygame.display.flip()   #Wyświetlenie nowej klatki

        fps = clock.get_fps()
        title = (f"Alg: {algorithm.upper()} | Boxes: {len(boxes)}  Collisions: {len(collisions)}  "
                 f"Time: {(t2-t1)*1000:.2f} ms  Checks: {checks}  FPS: {fps:.1f}")
        pygame.display.set_caption(title)
        clock.tick(30)  #Ograniczam symulacje do 30 FPS

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
