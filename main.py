"""视频动画播放控制 - 一体化入口（合并版）"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import os
import json
import socket
import threading
import time
import queue
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


def natural_sort_key(s):
    """自然排序键：将 'frame_10.png' 中的数字按数值排序而非字符串"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]



class PlayerState(Enum):
    IDLE = "idle"
    PLAYING_A = "playing_a"
    A_TO_B = "a_to_b"      # A播完，切换B的过渡瞬间
    PLAYING_B = "playing_b"
    STOPPED = "stopped"


@dataclass
class AnimationConfig:
    """序列帧动画配置"""
    path: str = ""
    loop: bool = False
    frame_duration: float = 0.1   # 每帧显示秒数
    total_frames: int = 0
    frame_files: list = field(default_factory=list)


class BackgroundVideoReader:
    """背景视频读取器 - 独立线程，持续循环"""
    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: str = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0

    def open(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        self._video_path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return True

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    def _read_loop(self):
        while self._running:
            try:
                cap = cv2.VideoCapture(self._video_path)
                if not cap.isOpened():
                    time.sleep(1)
                    continue
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        # 循环播放
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    with self._lock:
                        self._latest_frame = frame.copy()
                    # 非阻塞放入队列
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        # 丢弃旧帧
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                cap.release()
            except Exception as e:
                print(f"[VideoReader] 错误: {e}")
                time.sleep(1)

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧（非阻塞）"""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            return None

    def is_running(self) -> bool:
        return self._running


class SequenceAnimator:
    """序列帧动画播放器 - 按需加载帧，不预加载到内存"""
    def __init__(self):
        self.config: Optional[AnimationConfig] = None
        self._frame_files: list = []     # 文件路径列表（不加载到内存）
        self._current_idx: int = 0
        self._frame_timer: float = 0.0
        self._playing = False
        self._loop = False
        self._finished = False
        self._lock = threading.Lock()
        self._last_frame: Optional[Image.Image] = None
        self._cache: dict = {}           # 按需缓存：index -> PIL Image

    def load_config(self, cfg: AnimationConfig) -> bool:
        """加载序列帧配置（只扫描目录，不加载图片到内存）"""
        if not cfg.path or not os.path.isdir(cfg.path):
            return False
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        files = sorted([
            os.path.join(cfg.path, f)
            for f in os.listdir(cfg.path)
            if os.path.splitext(f)[1].lower() in exts
        ], key=natural_sort_key)
        if not files:
            return False
        cfg.total_frames = len(files)
        cfg.frame_files = files
        self.config = cfg
        self._loop = cfg.loop
        self._frame_files = files
        self._cache = {}
        self._last_frame = None
        print(f"[Animator] 已扫描 {len(files)} 帧（按需加载）: {cfg.path}")
        return True

    def _load_frame(self, idx: int) -> Optional[Image.Image]:
        """按需加载指定帧（带简单缓存）"""
        if idx < 0 or idx >= len(self._frame_files):
            return self._last_frame
        if idx in self._cache:
            return self._cache[idx]
        try:
            img = Image.open(self._frame_files[idx]).convert("RGBA")
            # 只缓存当前帧
            self._cache = {idx: img}
            return img
        except Exception as e:
            print(f"[Animator] 加载帧 {idx} 失败: {e}")
            return self._last_frame

    def start(self):
        """开始播放"""
        with self._lock:
            self._playing = True
            self._finished = False
            self._current_idx = 0
            self._frame_timer = 0.0

    def stop(self):
        """停止播放，保留最后一帧"""
        with self._lock:
            self._playing = False
            self._finished = True

    def reset(self):
        with self._lock:
            self._playing = False
            self._finished = False
            self._current_idx = 0
            self._frame_timer = 0.0
            self._frame_files = []
            self._cache = {}
            self._last_frame = None
            self.config = None

    def update(self, dt: float) -> bool:
        """更新动画状态，返回是否播放完成"""
        with self._lock:
            if not self._playing or not self._frame_files:
                return False

            self._frame_timer += dt
            frame_duration = self.config.frame_duration if self.config else 0.1

            if self._frame_timer >= frame_duration:
                self._frame_timer -= frame_duration
                self._current_idx += 1

                if self._current_idx >= len(self._frame_files):
                    if self._loop:
                        self._current_idx = 0
                    else:
                        self._current_idx = len(self._frame_files) - 1
                        self._playing = False
                        self._finished = True
                        return True  # 播放完成
            return False

    def get_current_frame(self) -> Optional[Image.Image]:
        """获取当前帧（按需加载）"""
        with self._lock:
            if not self._frame_files:
                return None
            frame = self._load_frame(self._current_idx)
            if frame is not None:
                self._last_frame = frame
            return frame

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def is_finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def current_idx(self) -> int:
        with self._lock:
            return self._current_idx

    @property
    def total_frames(self) -> int:
        with self._lock:
            return len(self._frame_files)

    @property
    def loop(self) -> bool:
        with self._lock:
            return self._loop


class StateMachine:
    """状态机：空闲 → 播放A → A完成 → 播放B → B完成 → 空闲"""
    def __init__(self):
        self.state = PlayerState.IDLE
        self._current_seq = "none"
        self._on_state_change: Optional[Callable] = None
        self._lock = threading.Lock()

    def set_callback(self, cb: Callable):
        self._on_state_change = cb

    def transition_to(self, new_state: PlayerState):
        with self._lock:
            old = self.state
            self.state = new_state
            if new_state == PlayerState.PLAYING_A:
                self._current_seq = "A"
            elif new_state == PlayerState.PLAYING_B:
                self._current_seq = "B"
            elif new_state == PlayerState.IDLE:
                self._current_seq = "none"
        if self._on_state_change:
            self._on_state_change(old, new_state)

    def get_state(self) -> PlayerState:
        with self._lock:
            return self.state

    def get_current_seq(self) -> str:
        with self._lock:
            return self._current_seq


class CommandHandler:
    """UDP指令接收器"""
    def __init__(self, port: int = 9999):
        self.port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self.on_command: Optional[Callable] = None
        self.on_log: Optional[Callable] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None

    def _listen_loop(self):
        while self._running:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._sock.settimeout(2.0)
                self._sock.bind(("0.0.0.0", self.port))
                self._log(f"UDP 指令监听端口: {self.port}")

                while self._running:
                    try:
                        data, addr = self._sock.recvfrom(65535)
                        msg = data.decode("utf-8", errors="replace").strip()
                        self._log(f"收到指令: {msg} (来自 {addr})")
                        if self.on_command:
                            self.on_command(msg)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    except Exception as e:
                        self._log(f"UDP异常: {e}")
                self._sock.close()
            except OSError as e:
                self._log(f"UDP绑定失败({self.port}): {e}")
                time.sleep(3)
            except Exception as e:
                self._log(f"UDP异常: {e}")
                time.sleep(3)

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)


class PlaybackEngine:
    """播放引擎 - 整合所有组件"""
    def __init__(self):
        self.video_reader = BackgroundVideoReader()
        self.animator_a = SequenceAnimator()
        self.animator_b = SequenceAnimator()
        self.state_machine = StateMachine()
        self.cmd_handler = CommandHandler()

        self._active_animator: Optional[SequenceAnimator] = None
        self._waiting_for_b = False
        self._lock = threading.Lock()

        # 回调
        self.on_log: Optional[Callable] = None
        self.on_state_change: Optional[Callable] = None

        # 设置状态机回调
        self.state_machine.set_callback(self._state_changed)

        # 设置指令回调
        self.cmd_handler.on_command = self._handle_command

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)

    def _state_changed(self, old: PlayerState, new: PlayerState):
        self._log(f"状态: {old.value} → {new.value}")
        if self.on_state_change:
            self.on_state_change(new)

    def _handle_command(self, msg: str):
        """解析UDP指令"""
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            self._log(f"指令解析失败: {msg}")
            return

        a_path = data.get("a_path", "")
        a_loop = data.get("a_loop", False)
        a_duration = data.get("a_frame_duration", 0.1)
        b_path = data.get("b_path", "")
        b_loop = data.get("b_loop", False)
        b_duration = data.get("b_frame_duration", 0.1)

        if not a_path or not b_path:
            self._log("指令缺少必要参数: a_path / b_path")
            return

        self._log(f"加载序列A: {a_path} (loop={a_loop}), 序列B: {b_path} (loop={b_loop})")

        # ⚠️ 先reset清除旧状态，再加载帧（顺序不能反）
        self.animator_a.reset()
        self.animator_b.reset()

        cfg_a = AnimationConfig(path=a_path, loop=a_loop, frame_duration=a_duration)
        cfg_b = AnimationConfig(path=b_path, loop=b_loop, frame_duration=b_duration)

        ok_a = self.animator_a.load_config(cfg_a)
        ok_b = self.animator_b.load_config(cfg_b)

        if not ok_a:
            self._log(f"序列A加载失败: {a_path}")
            return
        if not ok_b:
            self._log(f"序列B加载失败: {b_path}")
            return

        # 触发播放
        self._start_playback()

    def _start_playback(self):
        """开始播放A→B序列（帧已加载，只启动播放）"""
        self._waiting_for_b = False
        self.animator_a.start()
        self._active_animator = self.animator_a
        self.state_machine.transition_to(PlayerState.PLAYING_A)
        self._log(f"▶ 开始播放序列A ({self.animator_a.total_frames}帧, loop={self.animator_a.loop})")

    def update(self, dt: float):
        """每帧更新（由主循环调用）"""
        state = self.state_machine.get_state()

        if state == PlayerState.PLAYING_A:
            finished = self.animator_a.update(dt)
            if finished:
                self._log("✓ 序列A播放完成")
                # 立即切换B
                self.animator_b.start()
                self._active_animator = self.animator_b
                self.state_machine.transition_to(PlayerState.PLAYING_B)
                self._log(f"▶ 开始播放序列B ({self.animator_b.total_frames}帧, loop={self.animator_b.loop})")

        elif state == PlayerState.PLAYING_B:
            finished = self.animator_b.update(dt)
            if finished:
                self._log("✓ 序列B播放完成")
                self._active_animator = None
                self.state_machine.transition_to(PlayerState.IDLE)
                self._log("⏹ 播放结束，进入空闲状态")

    def get_current_animation_frame(self):
        """获取当前动画帧（PIL Image或None）"""
        state = self.state_machine.get_state()
        if state == PlayerState.PLAYING_A:
            return self.animator_a.get_current_frame()
        elif state == PlayerState.PLAYING_B:
            return self.animator_b.get_current_frame()
        return None

    def stop_current(self):
        """强制停止当前播放"""
        self.animator_a.stop()
        self.animator_b.stop()
        self._active_animator = None
        self._waiting_for_b = False
        self.state_machine.transition_to(PlayerState.IDLE)
        self._log("⏹ 强制停止播放")

    def start(self, video_path: str, udp_port: int = 9999):
        """启动引擎"""
        # 打开视频
        if not self.video_reader.open(video_path):
            self._log(f"无法打开视频: {video_path}")
            return False
        self._log(f"背景视频已加载: {video_path} ({self.video_reader.width}x{self.video_reader.height}, {self.video_reader.fps:.1f}fps)")

        # 启动视频播放
        self.video_reader.start()

        # 启动UDP监听
        self.cmd_handler.port = udp_port
        self.cmd_handler.start()

        return True

    def shutdown(self):
        self.cmd_handler.stop()
        self.video_reader.stop()
        self.animator_a.reset()
        self.animator_b.reset()


class VideoAnimationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("视频动画播放控制 v1.0")
        self.root.attributes("-fullscreen", True)
        self.root.minsize(800, 600)

        self.engine = PlaybackEngine()
        self.engine.on_log = self._on_log
        self.engine.on_state_change = self._on_state_change

        # 渲染相关
        self._render_running = False
        self._last_render_time = time.time()
        self._bg_image: Optional[ImageTk.PhotoImage] = None
        self._display_size = (800, 600)
        self._last_render_bg = None

        # 配置路径
        self._config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_config.json")
        self._load_config()

        # 渲染缓存
        self._last_anim_idx = -1
        self._last_render_size = (0, 0)
        self._cached_composite: Optional[Image.Image] = None
        self._cached_anim_resized: Optional[Image.Image] = None
        self._cached_anim_size = (0, 0)

        self._build_ui()
        self._log("程序已启动")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── UI ───

    def _build_ui(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="选择背景视频...", command=self._select_video, accelerator="Ctrl+O")
        file_menu.add_command(label="保存配置", command=self._save_config, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        ctrl_menu = tk.Menu(menubar, tearoff=0)
        ctrl_menu.add_command(label="全屏切换", command=self._toggle_fullscreen, accelerator="F11")
        ctrl_menu.add_command(label="停止播放", command=self._stop_playback)
        ctrl_menu.add_separator()
        ctrl_menu.add_command(label="模拟指令...", command=self._simulate_command)
        menubar.add_cascade(label="控制", menu=ctrl_menu)

        port_menu = tk.Menu(menubar, tearoff=0)
        port_menu.add_command(label="UDP端口设置...", command=self._set_udp_port)
        menubar.add_cascade(label="设置", menu=port_menu)
        self.root.config(menu=menubar)

        self.root.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.root.bind("<Control-o>", lambda e: self._select_video())
        self.root.bind("<Control-s>", lambda e: self._save_config())

        # ─── 顶部工具栏 ───
        toolbar = ttk.Frame(self.root, padding=3)
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="背景视频:").pack(side=tk.LEFT, padx=2)
        self.video_path_var = tk.StringVar(value=self._cfg.get("video_path", ""))
        self.video_path_label = ttk.Label(toolbar, textvariable=self.video_path_var,
                                           font=("", 9), foreground="gray")
        self.video_path_label.pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.state_var = tk.StringVar(value="空闲")
        self.state_label = ttk.Label(toolbar, textvariable=self.state_var,
                                     font=("", 10, "bold"))
        self.state_label.pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.anim_progress_var = tk.StringVar(value="")
        self.anim_progress_label = ttk.Label(toolbar, textvariable=self.anim_progress_var,
                                              font=("", 9), foreground="cyan")
        self.anim_progress_label.pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(toolbar, text="UDP端口:").pack(side=tk.LEFT, padx=2)
        self.port_var = tk.StringVar(value=str(self._cfg.get("udp_port", 9999)))
        self.port_label = ttk.Label(toolbar, textvariable=self.port_var, font=("", 10, "bold"))
        self.port_label.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="全屏", command=self._toggle_fullscreen).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="停止", command=self._stop_playback).pack(side=tk.LEFT, padx=2)

        # ─── 序列帧动画配置面板 ───
        anim_frame = ttk.LabelFrame(self.root, text="序列帧动画配置", padding=5)
        anim_frame.pack(fill=tk.X, padx=5, pady=2)

        # 序列A
        row_a = ttk.Frame(anim_frame)
        row_a.pack(fill=tk.X, pady=2)
        ttk.Label(row_a, text="序列A", width=6).pack(side=tk.LEFT)
        self.a_path_var = tk.StringVar()
        self.a_path_entry = ttk.Entry(row_a, textvariable=self.a_path_var, width=50)
        self.a_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(row_a, text="选帧文件", command=lambda: self._select_seq_file("a")).pack(side=tk.LEFT, padx=2)
        self.a_loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_a, text="循环", variable=self.a_loop_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(row_a, text="帧时长(秒):").pack(side=tk.LEFT, padx=2)
        self.a_dur_var = tk.StringVar(value="0.1")
        ttk.Entry(row_a, textvariable=self.a_dur_var, width=6).pack(side=tk.LEFT, padx=2)
        self.a_status_var = tk.StringVar(value="未加载")
        ttk.Label(row_a, textvariable=self.a_status_var, foreground="gray", width=12).pack(side=tk.LEFT, padx=5)
        self.a_btn_test = ttk.Button(row_a, text="▶ 播放", command=self._test_seq_a, width=6)
        self.a_btn_test.pack(side=tk.LEFT, padx=2)

        # 序列B
        row_b = ttk.Frame(anim_frame)
        row_b.pack(fill=tk.X, pady=2)
        ttk.Label(row_b, text="序列B", width=6).pack(side=tk.LEFT)
        self.b_path_var = tk.StringVar()
        self.b_path_entry = ttk.Entry(row_b, textvariable=self.b_path_var, width=50)
        self.b_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(row_b, text="选帧文件", command=lambda: self._select_seq_file("b")).pack(side=tk.LEFT, padx=2)
        self.b_loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_b, text="循环", variable=self.b_loop_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(row_b, text="帧时长(秒):").pack(side=tk.LEFT, padx=2)
        self.b_dur_var = tk.StringVar(value="0.1")
        ttk.Entry(row_b, textvariable=self.b_dur_var, width=6).pack(side=tk.LEFT, padx=2)
        self.b_status_var = tk.StringVar(value="未加载")
        ttk.Label(row_b, textvariable=self.b_status_var, foreground="gray", width=12).pack(side=tk.LEFT, padx=5)
        self.b_btn_test = ttk.Button(row_b, text="▶ 播放", command=self._test_seq_b, width=6)
        self.b_btn_test.pack(side=tk.LEFT, padx=2)

        # 操作按钮行
        row_btn = ttk.Frame(anim_frame)
        row_btn.pack(fill=tk.X, pady=3)
        ttk.Button(row_btn, text="▶ A→B顺序播放", command=self._trigger_ui_playback).pack(side=tk.LEFT, padx=5)
        ttk.Button(row_btn, text="■ 停止", command=self._stop_playback).pack(side=tk.LEFT, padx=5)
        ttk.Separator(row_btn, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(row_btn, text="或通过UDP发送JSON指令").pack(side=tk.LEFT, padx=2)
        ttk.Button(row_btn, text="模拟指令", command=self._simulate_command).pack(side=tk.LEFT, padx=5)

        # ─── 主渲染区域 ───
        self.render_frame = ttk.LabelFrame(self.root, text="画面预览", padding=2)
        self.render_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.canvas = tk.Canvas(self.render_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self._canvas_image_id = None

        # ─── 底部日志面板 ───
        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding=2)
        log_frame.pack(fill=tk.X, padx=5, pady=2, side=tk.BOTTOM)

        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X)
        ttk.Button(log_btn_frame, text="清空", command=self._clear_log).pack(side=tk.LEFT, padx=2)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        # ─── 状态栏 ───
        status_bar = ttk.Frame(self.root, padding=2)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_info = ttk.Label(status_bar, text="就绪 | 等待选择背景视频并启动")
        self.status_info.pack(side=tk.LEFT, padx=5)

        if self._cfg.get("video_path") and os.path.exists(self._cfg.get("video_path", "")):
            self.root.after(500, self._auto_start)

    def _auto_start(self):
        vpath = self._cfg["video_path"]
        if self.engine.start(vpath, int(self._cfg.get("udp_port", 9999))):
            self._start_rendering()
            self.status_info.config(text=f"运行中 | 背景: {os.path.basename(vpath)}")
            self._log(f"已自动启动，视频: {vpath}")
        else:
            self._log(f"自动启动失败，视频路径无效: {vpath}")

    # ─── 高性能渲染循环 ───

    def _start_rendering(self):
        if self._render_running:
            return
        self._render_running = True
        self._last_render_time = time.time()
        self._render_loop()

    def _render_loop(self):
        if not self._render_running:
            return
        try:
            now = time.time()
            dt = now - self._last_render_time
            self._last_render_time = now

            # 更新引擎状态（动画帧推进）
            self.engine.update(min(dt, 0.1))

            # 更新动画进度显示
            state = self.engine.state_machine.get_state()
            if state in (PlayerState.PLAYING_A, PlayerState.PLAYING_B):
                anim = self.engine.animator_a if state == PlayerState.PLAYING_A else self.engine.animator_b
                self.anim_progress_var.set(f"{anim.current_idx+1}/{anim.total_frames}")
            else:
                self.anim_progress_var.set("")

            # 判断是否需要刷新画面
            need_render = False
            active_state = self.engine.state_machine.get_state()
            is_anim_playing = active_state in (PlayerState.PLAYING_A, PlayerState.PLAYING_B)

            bg_frame = self.engine.video_reader.get_frame()
            if bg_frame is not None:
                self._last_render_bg = bg_frame
                need_render = True
            elif self._last_render_bg is not None and is_anim_playing:
                # 有缓存背景帧 + 动画播放中 → 持续刷新
                need_render = True
            elif is_anim_playing:
                # 无背景视频但有动画播放 → 黑底+动画也需要渲染
                need_render = True

            if need_render:
                self._render_frame(self._last_render_bg)

            # 动画播放时高频刷新（30fps），空闲时低频（10fps）省资源
            idle = self.engine.state_machine.get_state() == PlayerState.IDLE
            self.root.after(30 if not idle else 100, self._render_loop)
        except Exception as e:
            self._log(f"渲染循环异常: {e}")
            self.root.after(30, self._render_loop)

    def _render_frame(self, bg_frame: Optional[np.ndarray] = None):
        """渲染帧：无动画用PPM直显，有动画用PIL合成（缓存结果避免重复计算）"""
        try:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            self._display_size = (cw, ch)
            anim_frame = self.engine.get_current_animation_frame()
            state = self.engine.state_machine.get_state()
            is_anim = state in (PlayerState.PLAYING_A, PlayerState.PLAYING_B)

            if bg_frame is not None:
                # ── 有背景视频 ──
                h, w = bg_frame.shape[:2]
                ratio = min(cw / w, ch / h)
                if ratio < 1.0:
                    tw, th = int(w * ratio), int(h * ratio)
                    bg_resized = cv2.resize(bg_frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                else:
                    bg_resized = bg_frame
                    tw, th = w, h

                if is_anim and anim_frame is not None:
                    # 有动画 → PIL合成（缓存合成结果，避免每帧重复）
                    self._render_with_anim(bg_resized, anim_frame, tw, th)
                else:
                    # 无动画 → PPM直显（OpenCV自动BGR→RGB，颜色正确且速度快）
                    _, encoded = cv2.imencode('.ppm', bg_resized)
                    self._bg_image = tk.PhotoImage(data=encoded.tobytes())
            else:
                # ── 无背景视频 ──
                if is_anim and anim_frame is not None:
                    aw, ah = anim_frame.size
                    ratio = min(cw / aw, ch / ah)
                    if ratio < 1.0:
                        dw, dh = int(aw * ratio), int(ah * ratio)
                        display_img = anim_frame.resize((dw, dh), Image.BILINEAR).convert("RGB")
                    else:
                        display_img = anim_frame.convert("RGB")
                    self._bg_image = ImageTk.PhotoImage(display_img)
                else:
                    # 全黑
                    if self._canvas_image_id is not None:
                        self.canvas.delete(self._canvas_image_id)
                        self._canvas_image_id = None
                    return

            # 居中显示
            cx, cy = cw // 2, ch // 2
            if self._canvas_image_id is None:
                self._canvas_image_id = self.canvas.create_image(
                    cx, cy, image=self._bg_image, anchor=tk.CENTER
                )
            else:
                self.canvas.itemconfig(self._canvas_image_id, image=self._bg_image)

        except Exception as e:
            self._log(f"渲染异常: {e}")

    def _render_with_anim(self, bg_resized: np.ndarray, anim_frame: Image.Image, tw: int, th: int):
        """PIL合成动画帧（缓存结果，仅动画帧变化或画布大小变化时重算）"""
        # 取当前动画帧索引
        state = self.engine.state_machine.get_state()
        if state == PlayerState.PLAYING_A:
            anim_idx = self.engine.animator_a.current_idx
        else:
            anim_idx = self.engine.animator_b.current_idx

        cache_key = (anim_idx, tw, th)

        # 检查缓存是否命中
        if (cache_key == (self._last_anim_idx, self._last_render_size[0], self._last_render_size[1])
                and self._cached_composite is not None):
            self._bg_image = ImageTk.PhotoImage(self._cached_composite)
            return

        # 缓存未命中 → 重新合成
        bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
        bg_pil = Image.fromarray(bg_rgb)

        # 缩放动画帧到目标尺寸
        anim_resized = anim_frame.resize((tw, th), Image.BILINEAR)

        if anim_resized.mode == "RGBA":
            bg_rgba = bg_pil.convert("RGBA")
            composite = Image.alpha_composite(bg_rgba, anim_resized)
        else:
            composite = anim_resized.convert("RGBA")

        display_img = composite.convert("RGB")

        # 更新缓存
        self._last_anim_idx = anim_idx
        self._last_render_size = (tw, th)
        self._cached_composite = display_img

        self._bg_image = ImageTk.PhotoImage(display_img)

    def _on_canvas_resize(self, event):
        pass

    # ─── 状态回调 ───

    def _on_state_change(self, state: PlayerState):
        state_names = {
            PlayerState.IDLE: "空闲",
            PlayerState.PLAYING_A: "▶ 播放序列A",
            PlayerState.PLAYING_B: "▶ 播放序列B",
            PlayerState.STOPPED: "已停止",
        }
        name = state_names.get(state, state.value)
        self.root.after(0, lambda: self.state_var.set(name))

    def _on_log(self, msg: str):
        self.root.after(0, self._append_log, msg)

    def _append_log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _log(self, msg: str):
        self.engine._log(msg)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ─── 操作 ───

    def _select_video(self):
        path = filedialog.askopenfilename(
            title="选择背景视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self._cfg["video_path"] = path
        self.video_path_var.set(os.path.basename(path))
        self._save_config()

        self.engine.shutdown()
        self._render_running = False

        if self.engine.start(path, int(self._cfg.get("udp_port", 9999))):
            self._start_rendering()
            self.status_info.config(text=f"运行中 | 背景: {os.path.basename(path)}")
            self._log(f"已切换背景视频: {path}")
        else:
            messagebox.showerror("错误", f"无法打开视频文件:\n{path}")

    # ─── 序列帧：选文件自动识别文件夹 ───

    def _select_seq_file(self, which: str):
        """选一张序列帧图片 → 自动识别文件夹 + 加载全部帧并播放"""
        path = filedialog.askopenfilename(
            title=f"选择序列{which.upper()}中的任意一张帧图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        # 提取文件夹路径
        folder = os.path.dirname(path)
        if which == "a":
            self.a_path_var.set(folder)
            self._load_and_play_seq_a(folder)
        else:
            self.b_path_var.set(folder)
            self._load_and_play_seq_b(folder)

    def _select_seq_dir(self, which: str):
        """备选：选文件夹方式"""
        path = filedialog.askdirectory(title=f"选择序列{which.upper()}帧图片目录")
        if not path:
            return
        if which == "a":
            self.a_path_var.set(path)
            self._load_and_play_seq_a(path)
        else:
            self.b_path_var.set(path)
            self._load_and_play_seq_b(path)

    def _load_and_play_seq_a(self, path: str):
        """加载序列A并自动播放"""
        cfg = AnimationConfig(
            path=path,
            loop=self.a_loop_var.get(),
            frame_duration=float(self.a_dur_var.get() or "0.1")
        )
        self.engine.animator_a.reset()
        ok = self.engine.animator_a.load_config(cfg)
        if not ok:
            self.a_status_var.set("加载失败")
            self._log(f"序列A加载失败: {path}")
            return
        count = self.engine.animator_a.total_frames
        self.a_status_var.set(f"{count}帧")
        self._log(f"序列A已加载 {count} 帧: {path}")

        self.engine.animator_b.reset()
        self.engine.animator_a.start()
        self.engine.state_machine.transition_to(PlayerState.PLAYING_A)
        self._start_rendering()
        self.status_info.config(text=f"▶ 播放序列A | 无背景视频")
        self._log(f"▶ 自动播放序列A ({count}帧)")

    def _load_and_play_seq_b(self, path: str):
        """加载序列B并自动播放"""
        cfg = AnimationConfig(
            path=path,
            loop=self.b_loop_var.get(),
            frame_duration=float(self.b_dur_var.get() or "0.1")
        )
        self.engine.animator_b.reset()
        ok = self.engine.animator_b.load_config(cfg)
        if not ok:
            self.b_status_var.set("加载失败")
            self._log(f"序列B加载失败: {path}")
            return
        count = self.engine.animator_b.total_frames
        self.b_status_var.set(f"{count}帧")
        self._log(f"序列B已加载 {count} 帧: {path}")

        self.engine.animator_a.reset()
        self.engine.animator_b.start()
        self.engine.state_machine.transition_to(PlayerState.PLAYING_B)
        self._start_rendering()
        self.status_info.config(text=f"▶ 播放序列B | 无背景视频")
        self._log(f"▶ 自动播放序列B ({count}帧)")

    def _build_trigger_json(self) -> str:
        data = {
            "a_path": self.a_path_var.get().strip(),
            "a_loop": self.a_loop_var.get(),
            "a_frame_duration": float(self.a_dur_var.get() or "0.1"),
            "b_path": self.b_path_var.get().strip(),
            "b_loop": self.b_loop_var.get(),
            "b_frame_duration": float(self.b_dur_var.get() or "0.1"),
        }
        return json.dumps(data, ensure_ascii=False)

    def _trigger_ui_playback(self):
        a_path = self.a_path_var.get().strip()
        b_path = self.b_path_var.get().strip()
        if not a_path or not b_path:
            messagebox.showwarning("提示", "请先选择序列A和序列B的目录")
            return
        if not os.path.isdir(a_path):
            messagebox.showerror("错误", f"序列A目录不存在:\n{a_path}")
            return
        if not os.path.isdir(b_path):
            messagebox.showerror("错误", f"序列B目录不存在:\n{b_path}")
            return
        cmd = self._build_trigger_json()
        self._log(f"UI触发: {cmd}")
        self.engine._handle_command(cmd)
        self._start_rendering()
        self.status_info.config(text="▶ A→B顺序播放 | 无背景视频")

    def _test_seq_a(self):
        a_path = self.a_path_var.get().strip()
        if not a_path or not os.path.isdir(a_path):
            messagebox.showwarning("提示", '请先选择序列A目录（点"选帧文件"选一张图）')
            return
        self._load_and_play_seq_a(a_path)

    def _test_seq_b(self):
        b_path = self.b_path_var.get().strip()
        if not b_path or not os.path.isdir(b_path):
            messagebox.showwarning("提示", '请先选择序列B目录（点"选帧文件"选一张图）')
            return
        self._load_and_play_seq_b(b_path)

    def _stop_playback(self):
        self.engine.stop_current()
        self._log("手动停止播放")

    def _toggle_fullscreen(self):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def _simulate_command(self):
        dialog = SimulateDialog(self.root)
        if dialog.result:
            self.engine._handle_command(dialog.result)
            self._log(f"模拟指令: {dialog.result}")

    def _set_udp_port(self):
        dialog = PortDialog(self.root, self._cfg.get("udp_port", 9999))
        if dialog.result:
            self._cfg["udp_port"] = dialog.result
            self.port_var.set(str(dialog.result))
            self._save_config()
            self._log(f"UDP端口已更新为: {dialog.result}，重启后生效")
            messagebox.showinfo("提示", "端口已保存，重启程序后生效")

    # ─── 配置 ───

    def _load_config(self):
        self._cfg = {"video_path": "", "udp_port": 9999}
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self._cfg.update(loaded)
        except:
            pass

    def _save_config(self):
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"保存配置失败: {e}")

    def _on_close(self):
        self._render_running = False
        self.engine.shutdown()
        self._save_config()
        self.root.destroy()


# ═══════════════════════════════════════════
#  Simulate Dialog
# ═══════════════════════════════════════════

class SimulateDialog:
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("模拟UDP指令")
        self.dialog.geometry("550x250")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        frame = ttk.Frame(self.dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="JSON指令格式:").pack(anchor=tk.W)
        ttk.Label(frame, text='{"a_path":"C:/frames/seqA","a_loop":false,"a_frame_duration":0.1,'
                              '"b_path":"C:/frames/seqB","b_loop":true,"b_frame_duration":0.1}',
                  font=("Consolas", 9), foreground="gray").pack(anchor=tk.W, pady=2)

        ttk.Label(frame, text="指令内容:").pack(anchor=tk.W, pady=(10, 2))
        self.text = tk.Text(frame, height=6, font=("Consolas", 10))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.insert("1.0", '{\n  "a_path": "",\n  "a_loop": false,\n  "a_frame_duration": 0.1,\n  "b_path": "",\n  "b_loop": true,\n  "b_frame_duration": 0.1\n}')

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="发送", command=self._ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self._cancel).pack(side=tk.LEFT, padx=5)

        self.dialog.wait_window()

    def _ok(self):
        content = self.text.get("1.0", tk.END).strip()
        try:
            json.loads(content)
            self.result = content
            self.dialog.destroy()
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON格式错误", str(e))

    def _cancel(self):
        self.dialog.destroy()


# ═══════════════════════════════════════════
#  Port Dialog
# ═══════════════════════════════════════════

class PortDialog:
    def __init__(self, parent, current_port: int):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("UDP端口设置")
        self.dialog.geometry("300x120")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        frame = ttk.Frame(self.dialog, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="UDP监听端口:").pack(anchor=tk.W)
        self.port_var = tk.StringVar(value=str(current_port))
        ttk.Entry(frame, textvariable=self.port_var, width=10).pack(anchor=tk.W, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="确定", command=self._ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self._cancel).pack(side=tk.LEFT, padx=5)

        self.dialog.wait_window()

    def _ok(self):
        try:
            port = int(self.port_var.get())
            if 1024 <= port <= 65535:
                self.result = port
                self.dialog.destroy()
            else:
                messagebox.showerror("错误", "端口范围: 1024-65535")
        except ValueError:
            messagebox.showerror("错误", "请输入有效数字")

    def _cancel(self):
        self.dialog.destroy()

def main():
    root = tk.Tk()
    app = VideoAnimationApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()