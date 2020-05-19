import pygame, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial
from threading import Thread
import base64
import json
import struct

ser = serial.Serial('/dev/ttyUSB0', 250000, timeout=1)

def read_esp():
    while True:
        try:
            espData = ser.readline().decode('utf-8')
            print(espData, end=None)
        except:
            pass

# Thread(target=read_esp).start()

screen = pygame.display.set_mode((500,500))

draw_on = False
last_pos = (0, 0)
color = (255, 255, 255)
radius = 20

def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

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
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos,  radius)
            last_pos = e.pos
        pygame.display.flip()

except StopIteration:
    pass
finally:
    print(pygame.surfarray.array2d(screen))
    pygame.image.save(screen,"Test.jpeg")
    img = cv2.imread('Test.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation=cv2.INTER_AREA)
    cv2.imshow('resized', resized)
    data = np.asarray(resized)
    data = data.flatten()/255
    data = list(data)

    for i in range(784):
        n = data[i]
        n = struct.pack('f', n)
        ser.write(n)
        ser.flush()
    
    for i in range(11):
        espData = ser.readline().decode('utf-8')
        print(espData, end=None)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



pygame.quit()