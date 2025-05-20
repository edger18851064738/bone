import sys
from PyQt5.QtWidgets import QApplication
from gui import MineGUI

def main():
    """主程序入口"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = MineGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()