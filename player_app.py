"""Video Animation Player - 主窗口与渲染（v2 - 高性能渲染版）"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
import time
import cv2
from PIL import Image, ImageTk
import numpy as np

from engine import (
    PlaybackEngine, PlayerState, BackgroundVideoReader,
    SequenceAnimator, AnimationConfig, CommandHandler
)


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

            # 获取背景帧，无新帧时用缓存帧
            bg_frame = self.engine.video_reader.get_frame()
            if bg_frame is not None:
                self._last_render_bg = bg_frame
                self._render_frame(bg_frame)
            elif self._last_render_bg is not None:
                self._render_frame(self._last_render_bg)

            # 固定16ms间隔（~60fps上限），渲染多快就跑多快
            self.root.after(16, self._render_loop)
        except Exception as e:
            self._log(f"渲染循环异常: {e}")
            self.root.after(16, self._render_loop)

    def _render_frame(self, bg_frame: np.ndarray):
        """高性能渲染：OpenCV缩放 + PPM直显（无动画时绕过PIL）"""
        try:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            self._display_size = (cw, ch)

            # 用OpenCV缩放（比PIL快5-10倍）
            h, w = bg_frame.shape[:2]
            ratio = min(cw / w, ch / h)
            if ratio < 1.0:
                tw, th = int(w * ratio), int(h * ratio)
                bg_resized = cv2.resize(bg_frame, (tw, th), interpolation=cv2.INTER_LINEAR)
            else:
                bg_resized = bg_frame
                tw, th = w, h

            # 获取动画帧
            anim_frame = self.engine.get_current_animation_frame()

            if anim_frame is not None:
                # 有动画 → 用PIL合成
                bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
                bg_pil = Image.fromarray(bg_rgb)
                anim_resized = anim_frame.resize((tw, th), Image.BILINEAR)
                if anim_resized.mode == "RGBA":
                    bg_rgba = bg_pil.convert("RGBA")
                    composite = Image.alpha_composite(bg_rgba, anim_resized)
                else:
                    composite = anim_resized.convert("RGBA")
                composite = composite.convert("RGB")
                self._bg_image = ImageTk.PhotoImage(composite)
            else:
                # 无动画 → 用PPM直显（完全绕过PIL，性能提升巨大）
                bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
                _, encoded = cv2.imencode('.ppm', bg_rgb)
                self._bg_image = tk.PhotoImage(data=encoded.tobytes())

            # 显示
            cx, cy = cw // 2, ch // 2
            if self._canvas_image_id is None:
                self._canvas_image_id = self.canvas.create_image(
                    cx, cy, image=self._bg_image, anchor=tk.CENTER
                )
            else:
                self.canvas.itemconfig(self._canvas_image_id, image=self._bg_image)

        except Exception as e:
            self._log(f"渲染异常: {e}")

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
        from engine import AnimationConfig
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
        self._log(f"▶ 自动播放序列A ({count}帧)")

    def _load_and_play_seq_b(self, path: str):
        """加载序列B并自动播放"""
        from engine import AnimationConfig
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

    def _test_seq_a(self):
        a_path = self.a_path_var.get().strip()
        if not a_path or not os.path.isdir(a_path):
            messagebox.showwarning("提示", "请先选择序列A目录（点"选帧文件"选一张图）")
            return
        self._load_and_play_seq_a(a_path)

    def _test_seq_b(self):
        b_path = self.b_path_var.get().strip()
        if not b_path or not os.path.isdir(b_path):
            messagebox.showwarning("提示", "请先选择序列B目录（点"选帧文件"选一张图）")
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