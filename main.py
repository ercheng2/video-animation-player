"""视频动画播放控制 - 入口"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from player_app import VideoAnimationApp


def main():
    root = tk.Tk()
    app = VideoAnimationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()