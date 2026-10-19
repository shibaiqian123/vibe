"""Read-only input display; GUI callbacks enqueue main-thread commands only."""
import numpy as np


class RgbComparePanel:
    def __init__(self, viewer, control):
        self.viewer, self.control = viewer, control
        with viewer._server.gui.add_folder("RGB 对照"):
            self.mode = viewer._server.gui.add_dropdown(
                "输入模式", options=("normal", "gray", "shuffle"), initial_value=control.mode)
            self.apply = viewer._server.gui.add_button("应用并重新开始")
            self.status = viewer._server.gui.add_markdown("等待策略前向；Camera Feeds 是未干预的原图。")
            viewer._server.gui.add_markdown(f"启动 Checkpoint（对照期间不要热切换）: `{control.checkpoint}`\n\nSeed: `{control.seed}`")
            blank = np.zeros((63, 112, 3), dtype=np.uint8)
            self.raw = viewer._server.gui.add_image(blank, label="原始相机 RGB", format="png")
            self.used = viewer._server.gui.add_image(blank, label="策略实际输入 RGB", format="png")

        @self.apply.on_click
        def apply(_event):
            viewer.request_action("rgb_compare", ("rgb_compare", str(self.mode.value)))

    def update(self, env_idx):
        c = self.control
        if not c.matched():
            self.raw.visible = self.used.visible = False
            self.status.content = f"模式 `{c.mode}`：等待更新；尚无匹配的策略前向。"
            return
        f = c.policy_frame
        if env_idx >= f.raw.shape[0]:
            self.raw.visible = self.used.visible = False
            return
        self.raw.visible = self.used.visible = True
        self.raw.image = f.raw[env_idx].cpu().numpy()
        self.used.image = f.used[env_idx].cpu().numpy()
        clip = f.clips[env_idx] if f.clips else "未知"
        frame = f.frames[env_idx] if f.frames else "未知"
        src = f.sources[env_idx]
        source_clip = f.clips[src] if f.clips else "未知"
        self.status.content = (
            f"模式 `{c.mode}` · env `{env_idx}` · clip `{clip}` · frame `{frame}`\n\n"
            f"策略输入步 `{f.step}` · 配置版本 `{f.version}`\n\n"
            + (f"**错配图像来自 env `{src}` / clip `{source_clip}`**\n\n" if c.mode == "shuffle" else "")
            + "两张图均为该次策略前向的快照；Camera Feeds 是实时原图。"
        )
