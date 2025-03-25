import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class Character:
    def __init__(self, team, hp=20, attack=5):
        self.team = team  # 0:玩家队伍 1:敌方队伍
        self.hp = hp
        self.attack = attack
        self.movement = 5  # 移动力
        self.pos = QPoint(0, 0)

class BattleGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_size = 40
        self.selected_char = None
        self.characters = []
        self.current_team = 0
        self.move_range = []
        
        # 初始化测试角色
        self.characters.append(Character(0))
        self.characters[-1].pos = QPoint(2, 2)
        self.characters.append(Character(1))
        self.characters[-1].pos = QPoint(5, 5)

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_grid(painter)
        self.draw_characters(painter)
        self.draw_move_range(painter)

    def draw_grid(self, painter):
        painter.setPen(Qt.GlobalColor.darkGray)
        for x in range(0, self.width(), self.grid_size):
            for y in range(0, self.height(), self.grid_size):
                painter.drawRect(x, y, self.grid_size, self.grid_size)

    def draw_characters(self, painter):
        for char in self.characters:
            color = QColor(Qt.GlobalColor.blue) if char.team == 0 else QColor(Qt.GlobalColor.red)
            if char == self.selected_char:
                painter.setBrush(color.lighter(150))
            else:
                painter.setBrush(color)
            
            rect = QRect(
                char.pos.x() * self.grid_size,
                char.pos.y() * self.grid_size,
                self.grid_size,
                self.grid_size
            )
            painter.drawEllipse(rect)

    def draw_move_range(self, painter):
        if not self.selected_char:
            return
        
        painter.setBrush(QColor(100, 200, 100, 100))
        for pos in self.move_range:
            painter.drawRect(
                pos.x() * self.grid_size,
                pos.y() * self.grid_size,
                self.grid_size,
                self.grid_size
            )

    def mousePressEvent(self, event):
        grid_pos = QPoint(
            event.position().x() // self.grid_size,
            event.position().y() // self.grid_size
        )
        
        # 选择角色
        if not self.selected_char:
            for char in self.characters:
                if char.pos == grid_pos and char.team == self.current_team:
                    self.selected_char = char
                    self.calculate_move_range()
                    self.update()
                    break
        else:
            # 移动角色
            if grid_pos in self.move_range:
                self.selected_char.pos = grid_pos
                self.selected_char = None
                self.move_range = []
                self.current_team = 1 - self.current_team  # 切换回合
                QMessageBox.information(self, "回合结束", f"轮到{'玩家' if self.current_team ==0 else '敌方'}回合")
                self.update()

    def calculate_move_range(self):
        if not self.selected_char:
            return
        
        self.move_range = []
        start = self.selected_char.pos
        movement = self.selected_char.movement
        
        # 简单曼哈顿距离计算移动范围
        for dx in range(-movement, movement+1):
            for dy in range(-movement, movement+1):
                if abs(dx) + abs(dy) <= movement:
                    new_pos = QPoint(start.x()+dx, start.y()+dy)
                    if self.is_valid_position(new_pos):
                        self.move_range.append(new_pos)

    def is_valid_position(self, pos):
        return 0 <= pos.x() < 10 and 0 <= pos.y() < 10

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("战棋游戏")
        self.setFixedSize(600, 600)
        
        self.battle_grid = BattleGrid()
        self.setCentralWidget(self.battle_grid)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())