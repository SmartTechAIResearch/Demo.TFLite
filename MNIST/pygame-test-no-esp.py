import pygame
import cv2
import numpy as np
from threading import Thread
import struct
import io
from tensorflow.keras.models import load_model

model_name = 'emnist-model_augm.h5'
LABELS = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T']

# model_name = 'mnist-model_augm.h5'
# LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

model = load_model(model_name)

# Pygame
resolution = (500, 500)
color = (255, 255, 255)
radius = 20

running = True



def roundline(srf, color, start, end, radius=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)

def process_data(data):
    data = np.expand_dims(np.expand_dims(data, 0), -1)
    predictions = model.predict_proba(data)
    for i in range(len(LABELS)):
        print(f"[{LABELS[i]}]: {predictions[0][i]}")

    predicted_class = np.argmax(predictions[0], axis=-1)
    print(f"You drew: {LABELS[predicted_class]}\n")

def main():
    draw_on = False
    last_pos = (0, 0)
    screen = pygame.display.set_mode(resolution)


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