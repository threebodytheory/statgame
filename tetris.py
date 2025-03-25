import sys
import random
from PyQt6.QtWidgets import QMainWindow, QFrame, QApplication
from PyQt6.QtGui import QPainter, QColor, QKeyEvent
from PyQt6.QtCore import Qt, QBasicTimer

class Tetris(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.tboard = Board(self)
        self.setCentralWidget(self.tboard)
        self.tboard.start()
        self.resize(300, 600)
        self.setWindowTitle('俄罗斯方块')
        self.show()

class Board(QFrame):
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 22
    SPEED = 300

    def __init__(self, parent):
        super().__init__(parent)
        self.timer = QBasicTimer()
        self.isWaitingAfterLine = False
        self.curX = 0
        self.curY = 0
        self.numLinesRemoved = 0
        self.board = []
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.clearBoard()

    def shapeAt(self, x, y):
        return self.board[(y * Board.BOARD_WIDTH) + x]

    def setShapeAt(self, x, y, shape):
        self.board[(y * Board.BOARD_WIDTH) + x] = shape

    def clearBoard(self):
        self.board = [Tetrominoe.NoShape] * (Board.BOARD_HEIGHT * Board.BOARD_WIDTH)

    def start(self):
        if self.timer.isActive():
            return

        self.isWaitingAfterLine = False
        self.numLinesRemoved = 0
        self.clearBoard()
        self.newPiece()
        self.timer.start(Board.SPEED, self)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.contentsRect()
        boardTop = rect.bottom() - Board.BOARD_HEIGHT * self.squareHeight()

        for i in range(Board.BOARD_HEIGHT):
            for j in range(Board.BOARD_WIDTH):
                shape = self.shapeAt(j, Board.BOARD_HEIGHT - i - 1)
                if shape != Tetrominoe.NoShape:
                    self.drawSquare(painter,
                        rect.left() + j * self.squareWidth(),
                        boardTop + i * self.squareHeight(), shape)

        if self.curPiece.shape() != Tetrominoe.NoShape:
            for i in range(4):
                x = self.curX + self.curPiece.x(i)
                y = self.curY - self.curPiece.y(i)
                self.drawSquare(painter, rect.left() + x * self.squareWidth(),
                    boardTop + (Board.BOARD_HEIGHT - y - 1) * self.squareHeight(),
                    self.curPiece.shape())

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        
        if key == Qt.Key.Key_Escape.value:
            self.close()
        elif key == Qt.Key.Key_Left.value:
            self.tryMove(self.curPiece, self.curX - 1, self.curY)
        elif key == Qt.Key.Key_Right.value:
            self.tryMove(self.curPiece, self.curX + 1, self.curY)
        elif key == Qt.Key.Key_Down.value:
            self.tryMove(self.curPiece.rotateRight(), self.curX, self.curY)
        elif key == Qt.Key.Key_Up.value:
            self.tryMove(self.curPiece.rotateLeft(), self.curX, self.curY)
        elif key == Qt.Key.Key_Space.value:
            self.dropDown()
        elif key == Qt.Key.Key_D.value:
            self.oneLineDown()

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            if self.isWaitingAfterLine:
                self.isWaitingAfterLine = False
                self.newPiece()
            else:
                self.oneLineDown()
        else:
            super().timerEvent(event)

    def dropDown(self):
        newY = self.curY
        while newY > 0:
            if not self.tryMove(self.curPiece, self.curX, newY - 1):
                break
            newY -= 1
        self.pieceDropped()

    def oneLineDown(self):
        if not self.tryMove(self.curPiece, self.curX, self.curY - 1):
            self.pieceDropped()

    def pieceDropped(self):
        for i in range(4):
            x = self.curX + self.curPiece.x(i)
            y = self.curY - self.curPiece.y(i)
            self.setShapeAt(x, y, self.curPiece.shape())
        self.removeFullLines()
        if not self.isWaitingAfterLine:
            self.newPiece()

    def removeFullLines(self):
        numFullLines = 0
        rowsToRemove = []
        for i in range(Board.BOARD_HEIGHT):
            n = 0
            for j in range(Board.BOARD_WIDTH):
                if not self.shapeAt(j, i) == Tetrominoe.NoShape:
                    n += 1
            if n == Board.BOARD_WIDTH:
                rowsToRemove.append(i)
                numFullLines += 1

        if numFullLines > 0:
            for i in rowsToRemove:
                for k in range(i, Board.BOARD_HEIGHT):
                    for j in range(Board.BOARD_WIDTH):
                        self.setShapeAt(j, k, self.shapeAt(j, k + 1))
            self.numLinesRemoved += numFullLines
            self.isWaitingAfterLine = True
            self.curPiece.setShape(Tetrominoe.NoShape)
            self.update()

    def newPiece(self):
        self.curPiece = Shape()
        self.curPiece.setRandomShape()
        self.curX = Board.BOARD_WIDTH // 2 - 1
        self.curY = Board.BOARD_HEIGHT - 1 + self.curPiece.minY()
        if not self.tryMove(self.curPiece, self.curX, self.curY):
            self.curPiece.setShape(Tetrominoe.NoShape)
            self.timer.stop()
            self.update()

    def tryMove(self, newPiece, newX, newY):
        for i in range(4):
            x = newX + newPiece.x(i)
            y = newY - newPiece.y(i)
            if x < 0 or x >= Board.BOARD_WIDTH or y < 0 or y >= Board.BOARD_HEIGHT:
                return False
            if self.shapeAt(x, y) != Tetrominoe.NoShape:
                return False
        self.curPiece = newPiece
        self.curX = newX
        self.curY = newY
        self.update()
        return True

    def drawSquare(self, painter, x, y, shape):
        colorTable = [0x000000, 0xCC6666, 0x66CC66, 0x6666CC,
                      0xCCCC66, 0xCC66CC, 0x66CCCC, 0xDAAA00]
        color = QColor(colorTable[shape])
        painter.fillRect(x + 1, y + 1, self.squareWidth() - 2,
            self.squareHeight() - 2, color)

    def squareWidth(self):
        return self.contentsRect().width() // Board.BOARD_WIDTH

    def squareHeight(self):
        return self.contentsRect().height() // Board.BOARD_HEIGHT

class Tetrominoe:
    NoShape = 0
    ZShape = 1
    SShape = 2
    LineShape = 3
    TShape = 4
    SquareShape = 5
    LShape = 6
    MirroredLShape = 7

class Shape:
    coordsTable = (
        ((0, 0), (0, 0), (0, 0), (0, 0)),
        ((0, -1), (0, 0), (-1, 0), (-1, 1)),
        ((0, -1), (0, 0), (1, 0), (1, 1)),
        ((0, -1), (0, 0), (0, 1), (0, 2)),
        ((-1, 0), (0, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((-1, -1), (0, -1), (0, 0), (0, 1)),
        ((1, -1), (0, -1), (0, 0), (0, 1))
    )

    def __init__(self):
        self.coords = [[0,0] for _ in range(4)]
        self.pieceShape = Tetrominoe.NoShape
        self.setShape(Tetrominoe.NoShape)

    def shape(self):
        return self.pieceShape

    def setShape(self, shape):
        table = Shape.coordsTable[shape]
        for i in range(4):
            for j in range(2):
                self.coords[i][j] = table[i][j]
        self.pieceShape = shape

    def setRandomShape(self):
        self.setShape(random.randint(1, 7))

    def x(self, index):
        return self.coords[index][0]

    def y(self, index):
        return self.coords[index][1]

    def minX(self):
        m = self.coords[0][0]
        for i in range(4):
            m = min(m, self.coords[i][0])
        return m

    def maxX(self):
        m = self.coords[0][0]
        for i in range(4):
            m = max(m, self.coords[i][0])
        return m

    def minY(self):
        m = self.coords[0][1]
        for i in range(4):
            m = min(m, self.coords[i][1])
        return m

    def maxY(self):
        m = self.coords[0][1]
        for i in range(4):
            m = max(m, self.coords[i][1])
        return m

    def rotateLeft(self):
        if self.pieceShape == Tetrominoe.SquareShape:
            return self
        result = Shape()
        result.pieceShape = self.pieceShape
        for i in range(4):
            result.coords[i][0] = self.coords[i][1]
            result.coords[i][1] = -self.coords[i][0]
        return result

    def rotateRight(self):
        if self.pieceShape == Tetrominoe.SquareShape:
            return self
        result = Shape()
        result.pieceShape = self.pieceShape
        for i in range(4):
            result.coords[i][0] = -self.coords[i][1]
            result.coords[i][1] = self.coords[i][0]
        return result

if __name__ == '__main__':
    app = QApplication([])
    tetris = Tetris()
    sys.exit(app.exec()) 