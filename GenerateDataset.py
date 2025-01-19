import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import random


def generateData(images):

    def createCanvas(img):
        x, y = random.randint(0, 100), random.randint(0, 100)
        x_max = x + 28
        y_max = y + 28
        canvas = np.zeros((128, 128))
        canvas[x:x_max, y:y_max] = img

        return canvas, [x, y, x_max, y_max]

    def saveImages(img, name):
        os.makedirs('Images', exist_ok=True)
        fig = plt.figure(frameon=False)
        plt.matshow(img)
        plt.savefig(f'Images/{name}.png')
        plt.close(fig)

    def saveBbox(points, name):
        os.makedirs('label', exist_ok=True)
        with open(f'label/{name}.txt', 'w') as f:
            f.write(f'''{points[0]} {points[1]}\n{points[0] + 28} {points[1]}\n
                        {points[0]} {points[1] + 28}\n{points[0] + 28} {points[1] + 28}''')
            
                    
    for i in range(len(images)):
        img, points = createCanvas(images[i].reshape(28, 28))
        saveImages(img, i)
        saveBbox(points, i)


    print('Successfully generated dataset, Count:', len(images))



if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    generateData(df.drop('label', axis=1).values)
