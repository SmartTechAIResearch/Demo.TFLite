import pygame
import cv2
import numpy as np
import serial
from threading import Thread
import struct
import io

# Serial
port = '/dev/ttyUSB0'
baud_rate = 250000

# Pygame
resolution = (500, 500)
color = (255, 255, 255)
radius = 20

running = True


def read_esp(ser):
    while running:
        try:
            line = ser.readline().decode('utf-8')[:-2]
            if line == "": continue
            print(line, end=None)
        except:
            pass
    ser.close()


def roundline(srf, color, start, end, radius=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def process_data(data):
    data = np.asarray(data)
    data_list = list(data.flatten() / 255)
    data_bytes = bytearray()

    for i in range(784):
        n = data_list[i]
        n = struct.pack('f', n)
        for b in n:
            data_bytes.append(b)
    
    return data_bytes


def send_data(ser, data):
    # print(f'Data: {data_bytes}')
    print(f'Length data sent: {len(data)}')
    ser.write(data)
    ser.flush()


def main():
    draw_on = False
    last_pos = (0, 0)
    screen = pygame.display.set_mode(resolution)

    # Initiate serial connection
    ser = serial.Serial(port, baud_rate, timeout=1)

    # Start reading incomming serial messages
    Thread(target=read_esp, args=(ser,)).start()

    # Main loop for drawing and sending data
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
                data = process_data(resized)
                send_data(ser, data)
            elif e.key == pygame.K_q:
                raise StopIteration
        pygame.display.flip()


if __name__ == '__main__':
    try:
        main()
    except StopIteration:
        pass
    finally:
        running = False
        pygame.quit()