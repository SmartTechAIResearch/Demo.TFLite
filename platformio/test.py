import pygame
import cv2
import numpy as np
import serial
from threading import Thread
import struct
import io

ser = serial.Serial('/dev/ttyUSB0', 250000, timeout=1)

running = True

def read_esp():
    while running:
        try:
            esp_data = ser.readline().decode('utf-8')[:-2]
            if esp_data == "": continue
            print(esp_data, end=None)
        except:
            pass
    ser.close()

Thread(target=read_esp).start()

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
        x = int(start[0]+float(i)/distance*dx)
        y = int(start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

if __name__ == '__main__':
    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            elif e.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.circle(screen, color, e.pos, radius)
                draw_on = True
            elif e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
            elif e.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, color, e.pos, radius)
                    roundline(screen, color, e.pos, last_pos,  radius)
                last_pos = e.pos
            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_c:
                    screen.fill((0,0,0))
                elif e.key == pygame.K_s or e.key == pygame.K_RETURN:
                    view = pygame.surfarray.array3d(screen)
                    view = view.transpose([1, 0, 2])
                    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (28,28), interpolation=cv2.INTER_AREA)
                    data = np.asarray(resized)
                    data_list = list(data.flatten()/255)
                    data_bytes = bytearray()

                    for i in range(784):
                        n = data_list[i]
                        n = struct.pack('f', n)
                        for b in n:
                            data_bytes.append(b)
                    
                    # print(f'Data: {data_bytes}')
                    print(f'Length data sent: {len(data_bytes)}')
                    ser.write(data_bytes)
                    ser.flush()
                elif e.key == pygame.K_q:
                    raise StopIteration
            pygame.display.flip()
    except StopIteration:
        pass
    finally:
        running = False
        pygame.quit()