import serial
from threading import Thread
import time
import numpy as np
import pygame 
import cv2

def serial_output():
    while running:
        try:
            espData = ser.readline().decode('utf-8')[:-2]
            if espData != "": print(espData, end=None)
        except:
            pass


def get_digit(img):
    #img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
    return resized

def processed_data(digit):
    data = np.asarray(digit)
    data = data.flatten()
    lst_data = []
    for i in data:
        lst_data.append(i)

    data = str(lst_data).replace(', ', ' ')
    data = data[1:-1]
    b = bytearray(data, 'utf-8')
    return b

def predict_digit(file):
    # digit = get_digit(file)
    view = pygame.surfarray.array3d(screen)
    view = view.transpose([1, 0, 2])
    img = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    data = processed_data(resized)
    print("Sending data....")
    ser.write(data)

def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

if __name__ == "__main__":
    running = True
    ser = serial.Serial('com4',256_000, timeout=1)
    Thread(target=serial_output).start()
    time.sleep(1)
    print("Initializing pygame")
    print("Press <C> to clear output, <P> to predict what digit it is and <Q> to quit.")
    screen = pygame.display.set_mode((800,800))
    digitfile =  "digit.jpeg"

    draw_on = False
    last_pos = (0, 0)
    color = (255, 255, 255)
    radius = 30

    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            if e.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.circle(screen, color, e.pos, radius)
                draw_on = True
            if e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_p:
                    pygame.image.save(screen,digitfile)
                    predict_digit(digitfile)
                elif e.key == pygame.K_c:
                    screen.fill((0,0,0))
                    pygame.display.update()
                elif e.key == pygame.K_q or e.key == pygame.K_a:
                    raise StopIteration
            if e.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, color, e.pos, radius)
                    roundline(screen, color, e.pos, last_pos,  radius)
                last_pos = e.pos
            pygame.display.flip()

    except StopIteration:
        pass
    finally:
        running = False
        pygame.quit()
       
