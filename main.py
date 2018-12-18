from racetrack import Racetrack
import pygame
import sys
import argparse
import time
import LAO_search
import BLAO_search

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
        bidirectional = False
        path_end = None
        if bidirectional:
            path, path_end = BLAO_search.search(self.track, search_method)
        else:
            path = LAO_search.search(self.track, search_method)

        pygame.init()
        self.displaySurface = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)
        self.displaySurface.fill((255, 255, 255))
        pygame.display.flip()
        pygame.display.set_caption(self.windowTitle)

        print("Results")
        #print("Path Length:", len(path))
        #print("States Explored:", statesExplored)

        self.drawExplored(path, set())
        self.drawPolicy(path, set())
        self.drawPolicy(path_end, set(), True)
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

    def drawPolicy(self, path, seen_nodes: set, invert=False, optimum=True):
        if path is None:
            return
        seen_nodes.add(path.state[:2].tobytes())
        p = path
        if invert:
            if optimum:
                color = (128, 0, 255)
            else:
                color = (128, 0, 0)
        else:
            if optimum:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

        self.drawArrow(p.state[0], p.state[1], p.state[2], p.state[3], color)

        actions = p.get_recommended_actions()
        if actions is not None:
            success, fail = actions
            if success.state[:2].tobytes() not in seen_nodes:
                seen_nodes = self.drawPolicy(success, seen_nodes, invert, optimum)
            optimum = False
            if fail.state[:2].tobytes() not in seen_nodes:
                seen_nodes = self.drawPolicy(fail, seen_nodes, invert, optimum)
        return seen_nodes

    def drawExplored(self, path, seen_nodes: set):
        if path is None:
            return
        seen_nodes.add(path.state[:2].tobytes())
        p = path
        color = (30, 30, 30)

        self.drawArrow(p.state[0], p.state[1], p.state[2], p.state[3], color)

        for actions in p.children:
            if actions is not None:
                success, fail = actions
                if success.state[:2].tobytes() not in seen_nodes:
                    seen_nodes = self.drawExplored(success, seen_nodes)
                if fail.state[:2].tobytes() not in seen_nodes:
                    seen_nodes = self.drawExplored(fail, seen_nodes)
        return seen_nodes

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
        for state in self.track.get_start():
            pygame.draw.rect(self.displaySurface, (0, 0, 255), (state[1] * self.blockSizeX + self.blockSizeX / 4, state[0] * self.blockSizeY + self.blockSizeY / 4, self.blockSizeX * 0.5, self.blockSizeY * 0.5), 0)

    # Draws arrow
    def drawArrow(self, row, col, row_accel, col_accel, color):
        pygame.draw.rect(self.displaySurface, color, (col * self.blockSizeX + self.blockSizeX / 4, row * self.blockSizeY + self.blockSizeY / 4, self.blockSizeX * 0.5, self.blockSizeY * 0.5), 0)

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
