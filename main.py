from racetrack import Racetrack
import pygame
import sys
import argparse
import time
import search

import pygame.locals



class Application:
    def __init__(self, filename: str, scale=20, fps=30):
        self.running = True
        self.displaySurface = None
        self.scale = scale
        self.fps = fps
        self.windowTitle = "Racetrack MDP: " + filename

        self.track = Racetrack(filename)
        self.dim = self.track.get_dimensions()

        self.windowHeight = self.dim[0] * self.scale
        self.windowWidth = self.dim[1] * self.scale

        self.blockSizeX = int(self.windowWidth / self.dim[1])
        self.blockSizeY = int(self.windowHeight / self.dim[0])

    def execute(self, search_method: str, save: str):

        path, statesExplored = search.search(self.track, search_method)

        pygame.init()
        self.displaySurface = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)
        self.displaySurface.fill((255, 255, 255))
        pygame.display.flip()
        pygame.display.set_caption(self.windowTitle)

        print("Results")
        #print("Path Length:", len(path))
        #print("States Explored:", statesExplored)
        #self.drawPolicy(path)

        self.drawTrack()
        self.drawStart()
        self.drawObjective()

        pygame.display.flip()
        if save is not None:
            pygame.image.save(self.displaySurface, save)
            self.running = False

        clock = pygame.time.Clock()

        while self.running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            clock.tick(self.fps)

            if any(keys):
                raise SystemExit

    def getColor(self, pathLength, index):
        step = 255 / pathLength
        green = index * step
        red = 255 - green
        return red, green, 0

    def drawPolicy(self, path):
        for p in range(len(path)):
            color = self.getColor(len(path), p)
            self.drawSquare(path[p][0], path[p][1], color)

    # Simple wrapper for drawing a wall as a rectangle
    def drawWall(self, row, col):
        pygame.draw.rect(self.displaySurface, (0, 0, 0), (col * self.blockSizeX, row * self.blockSizeY, self.blockSizeX, self.blockSizeY), 0)

    # Simple wrapper for drawing a circle
    def drawCircle(self, row, col, color, radius=None):
        if radius is None:
            radius = min(self.blockSizeX, self.blockSizeY) / 4
        pygame.draw.circle(self.displaySurface, color, (int(col * self.blockSizeX + self.blockSizeX / 2), int(row * self.blockSizeY + self.blockSizeY / 2)), int(radius))

    def drawSquare(self, row, col, color):
        pygame.draw.rect(self.displaySurface, color, (col * self.blockSizeX, row * self.blockSizeY, self.blockSizeX, self.blockSizeY), 0)

    # Draws the objectives to the display context
    def drawObjective(self):
        for obj in self.track.get_objectives():
            self.drawCircle(obj[0], obj[1], (0, 0, 0))

    # Draws start location of path
    def drawStart(self):
        for x, y in self.track.get_start():
            pygame.draw.rect(self.displaySurface, (0, 0, 255), (y * self.blockSizeX + self.blockSizeX / 4, x * self.blockSizeY + self.blockSizeY / 4, self.blockSizeX * 0.5, self.blockSizeY * 0.5), 0)

    # Draws the full maze to the display context
    def drawTrack(self):
        for row in range(self.dim[0]):
            for col in range(self.dim[1]):
                if self.track.is_wall(row, col):
                    self.drawWall(row, col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', default="small_track.txt", type=str,
                        help='path to racetrack file - default ')
    parser.add_argument('--method', dest="search", type=str, default="dblao",
                        choices=["lao", "blao", "dblao"],
                        help='search method - default dblao')
    parser.add_argument('--scale', dest="scale", type=int, default=20,
                        help='scale - default: 20')
    parser.add_argument('--fps', dest="fps", type=int, default=30,
                        help='fps for the display - default 30')
    parser.add_argument('--save', dest="save", type=str, default=None,
                        help='save output to image file - default not saved')

    args = parser.parse_args()
    app = Application(args.filename, args.scale, args.fps)
    app.execute(args.search, args.save)
