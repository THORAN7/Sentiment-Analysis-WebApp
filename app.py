import math
import tkinter as tk
from tkinter import messagebox

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


class PremiumSentimentApp:
    BG_TOP = "#0b1220"
    BG_MID = "#111b2e"
    BG_BOTTOM = "#1a2740"
    CARD_BG = "#0f1728"
    CARD_EDGE = "#22314c"
    INPUT_BG = "#0b1322"
    TEXT_PRIMARY = "#f8fafc"
    TEXT_SECONDARY = "#cbd5e1"
    TEXT_MUTED = "#94a3b8"
    ACCENT = "#60a5fa"
    SUCCESS = "#22c55e"
    ERROR = "#ef4444"
    NEUTRAL = "#a78bfa"

    FONT_H1 = ("Segoe UI Variable Display", 24, "bold")
    FONT_H2 = ("Segoe UI Variable Text", 12, "bold")
    FONT_BODY = ("Segoe UI Variable Text", 11)
    FONT_BODY_BOLD = ("Segoe UI Variable Text", 13, "bold")
    FONT_SMALL = ("Segoe UI Variable Text", 9)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sentiment Analysis App")
        self.root.geometry("900x640")
        self.root.minsize(820, 600)
        self.root.configure(bg=self.BG_TOP)

        self.sia = self._setup_sentiment_engine()

        self.canvas = tk.Canvas(self.root, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.card = tk.Frame(
            self.canvas,
            bg=self.CARD_BG,
            highlightthickness=1,
            highlightbackground=self.CARD_EDGE,
        )
        self.card_window = self.canvas.create_window(0, 0, window=self.card, anchor="center")

        self.card_width = 680
        self.card_height = 450

        self.orbs = self._create_orbs()
        self.phase = 0.0

        self._build_ui()
        self._bind_events()
        self._layout_card()
        self._animate_background()

    def _setup_sentiment_engine(self):
        try:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:
            messagebox.showwarning(
                "NLTK setup warning",
                f"Sentiment engine could not be initialized.\n\n{exc}",
            )
            return None

    @staticmethod
    def _hex_to_rgb(color: str):
        value = color.lstrip("#")
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _blend(self, a: str, b: str, ratio: float):
        ratio = max(0.0, min(1.0, ratio))
        ar, ag, ab = self._hex_to_rgb(a)
        br, bg, bb = self._hex_to_rgb(b)
        return self._rgb_to_hex(
            (
                round(ar + (br - ar) * ratio),
                round(ag + (bg - ag) * ratio),
                round(ab + (bb - ab) * ratio),
            )
        )

    def _paint_gradient(self, width: int, height: int):
        self.canvas.delete("gradient")
        steps = max(height, 1)
        for y in range(steps):
            ratio = y / steps
            if ratio < 0.5:
                color = self._blend(self.BG_TOP, self.BG_MID, ratio / 0.5)
            else:
                color = self._blend(self.BG_MID, self.BG_BOTTOM, (ratio - 0.5) / 0.5)
            self.canvas.create_line(0, y, width, y, fill=color, tags=("gradient",))

    def _create_orbs(self):
        specs = [
            {"x": 110, "y": 120, "r": 65, "dx": 0.14, "dy": 0.10, "color": "#38bdf8"},
            {"x": 730, "y": 155, "r": 80, "dx": -0.10, "dy": 0.08, "color": "#f472b6"},
            {"x": 760, "y": 470, "r": 55, "dx": -0.08, "dy": -0.11, "color": "#f59e0b"},
            {"x": 180, "y": 505, "r": 72, "dx": 0.11, "dy": -0.09, "color": "#a78bfa"},
        ]
        for orb in specs:
            orb["item"] = self.canvas.create_oval(
                orb["x"] - orb["r"],
                orb["y"] - orb["r"],
                orb["x"] + orb["r"],
                orb["y"] + orb["r"],
                fill=orb["color"],
                outline="",
                stipple="gray50",
                tags=("orb",),
            )
        return specs

    def _layout_card(self, _event=None):
        width = max(self.root.winfo_width(), 1)
        height = max(self.root.winfo_height(), 1)
        self._paint_gradient(width, height)
        self.canvas.coords(self.card_window, width / 2, height / 2)
        self.canvas.itemconfigure(self.card_window, width=self.card_width, height=self.card_height)

    def _animate_background(self, *_args):
        self.phase += 0.02
        width = max(self.root.winfo_width(), 1)
        height = max(self.root.winfo_height(), 1)

        for orb in self.orbs:
            x1, y1, x2, y2 = self.canvas.coords(orb["item"])

            if x1 <= 0 or x2 >= width:
                orb["dx"] *= -1
            if y1 <= 0 or y2 >= height:
                orb["dy"] *= -1

            dx = orb["dx"] + math.sin(self.phase + x1 / 150) * 0.05
            dy = orb["dy"] + math.cos(self.phase + y1 / 160) * 0.04
            self.canvas.move(orb["item"], dx, dy)

        self.root.after(40, self._animate_background, None)

    def _build_ui(self):
        header = tk.Frame(self.card, bg=self.CARD_BG)
        header.pack(fill="x", padx=32, pady=(26, 10))

        tk.Label(
            header,
            text="Sentiment Analyzer",
            font=self.FONT_H1,
            fg=self.ACCENT,
            bg=self.CARD_BG,
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Professional desktop interface with clear sentiment insights.",
            font=self.FONT_BODY,
            fg=self.TEXT_SECONDARY,
            bg=self.CARD_BG,
        ).pack(anchor="w", pady=(4, 0))

        tk.Frame(self.card, bg=self.CARD_EDGE, height=1).pack(fill="x", padx=32, pady=(4, 18))

        tk.Label(
            self.card,
            text="Text Input",
            font=self.FONT_H2,
            fg=self.TEXT_PRIMARY,
            bg=self.CARD_BG,
        ).pack(anchor="w", padx=32)

        tk.Label(
            self.card,
            text="Type any sentence, then click Analyze Sentiment.",
            font=self.FONT_SMALL,
            fg=self.TEXT_MUTED,
            bg=self.CARD_BG,
        ).pack(anchor="w", padx=32, pady=(3, 0))

        entry_wrap = tk.Frame(
            self.card,
            bg=self.INPUT_BG,
            highlightthickness=1,
            highlightbackground="#2e3f5d",
        )
        entry_wrap.pack(fill="x", padx=32, pady=(10, 14))

        self.text_entry = tk.Entry(
            entry_wrap,
            font=self.FONT_BODY,
            fg=self.TEXT_PRIMARY,
            bg=self.INPUT_BG,
            insertbackground=self.TEXT_PRIMARY,
            relief="flat",
            highlightthickness=0,
        )
        self.text_entry.pack(fill="x", padx=14, pady=14)
        self.text_entry.focus_set()

        action_row = tk.Frame(self.card, bg=self.CARD_BG)
        action_row.pack(fill="x", padx=32, pady=(0, 12))

        self.analyze_btn = tk.Button(
            action_row,
            text="Analyze Sentiment",
            font=self.FONT_H2,
            command=self.analyze_sentiment,
            fg="white",
            bg=self.ACCENT,
            activeforeground="white",
            activebackground=self.ACCENT,
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=9,
        )
        self.analyze_btn.pack(side="left")
        self.analyze_btn.bind("<Enter>", lambda _e: self.analyze_btn.config(bg=self._blend(self.ACCENT, "#ffffff", 0.08)))
        self.analyze_btn.bind("<Leave>", lambda _e: self.analyze_btn.config(bg=self.ACCENT))

        tk.Label(
            action_row,
            text="Press Enter to analyze",
            font=self.FONT_SMALL,
            fg=self.TEXT_MUTED,
            bg=self.CARD_BG,
        ).pack(side="left", padx=14)

        self.result_card = tk.Frame(
            self.card,
            bg=self.INPUT_BG,
            highlightthickness=1,
            highlightbackground=self.ACCENT,
        )
        self.result_card.pack(fill="both", expand=True, padx=32, pady=(0, 24))

        top_row = tk.Frame(self.result_card, bg=self.INPUT_BG)
        top_row.pack(fill="x", padx=18, pady=(16, 8))

        self.status_title = tk.StringVar(value="Sentiment Summary")
        self.status_chip = tk.StringVar(value="Idle")

        self.status_title_label = tk.Label(
            top_row,
            textvariable=self.status_title,
            font=self.FONT_H2,
            fg=self.TEXT_SECONDARY,
            bg=self.INPUT_BG,
        )
        self.status_title_label.pack(side="left")

        self.status_chip_label = tk.Label(
            top_row,
            textvariable=self.status_chip,
            font=self.FONT_SMALL,
            fg="#0b1322",
            bg=self.ACCENT,
            padx=10,
            pady=3,
        )
        self.status_chip_label.pack(side="right")

        self.primary_result = tk.StringVar(value="Ready for analysis")
        tk.Label(
            self.result_card,
            textvariable=self.primary_result,
            font=self.FONT_BODY_BOLD,
            fg=self.TEXT_PRIMARY,
            bg=self.INPUT_BG,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=18, pady=(0, 10))

        metrics = tk.Frame(self.result_card, bg=self.INPUT_BG)
        metrics.pack(fill="x", padx=18, pady=(0, 10))

        self.metric_sentiment = tk.StringVar(value="Waiting")
        self.metric_compound = tk.StringVar(value="-")
        self.metric_confidence = tk.StringVar(value="-")

        self._metric_tile(metrics, "Sentiment", self.metric_sentiment).pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._metric_tile(metrics, "Compound", self.metric_compound).pack(side="left", fill="x", expand=True, padx=3)
        self._metric_tile(metrics, "Confidence", self.metric_confidence).pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.bar_canvas = tk.Canvas(self.result_card, height=14, bg=self.INPUT_BG, bd=0, highlightthickness=0)
        self.bar_canvas.pack(fill="x", padx=18, pady=(2, 10))
        self.bar_bg = self.bar_canvas.create_rectangle(0, 0, 1, 14, fill="#1c2a43", outline="")
        self.bar_fill = self.bar_canvas.create_rectangle(0, 0, 1, 14, fill=self.ACCENT, outline="")

        tk.Label(
            self.result_card,
            text="Details",
            font=self.FONT_H2,
            fg=self.TEXT_SECONDARY,
            bg=self.INPUT_BG,
        ).pack(anchor="w", padx=18, pady=(0, 2))

        self.detail_text = tk.StringVar(value="Sentiment details will appear here after analysis.")
        tk.Label(
            self.result_card,
            textvariable=self.detail_text,
            font=self.FONT_BODY,
            fg=self.TEXT_MUTED,
            bg=self.INPUT_BG,
            justify="left",
            anchor="w",
            wraplength=620,
        ).pack(fill="x", padx=18, pady=(0, 14))

    def _metric_tile(self, parent: tk.Frame, title: str, value_var: tk.StringVar):
        tile = tk.Frame(parent, bg="#121f35", highlightthickness=1, highlightbackground="#24344f")
        tk.Label(tile, text=title, font=self.FONT_SMALL, fg=self.TEXT_MUTED, bg="#121f35").pack(anchor="w", padx=10, pady=(7, 0))
        tk.Label(tile, textvariable=value_var, font=self.FONT_BODY_BOLD, fg=self.TEXT_PRIMARY, bg="#121f35").pack(anchor="w", padx=10, pady=(2, 8))
        return tile

    def _bind_events(self):
        self.root.bind("<Configure>", self._layout_card)
        self.root.bind("<Return>", lambda _e: self.analyze_sentiment())

    def _update_strength_bar(self, compound: float, fill_color: str):
        self.bar_canvas.update_idletasks()
        width = max(self.bar_canvas.winfo_width(), 1)
        self.bar_canvas.coords(self.bar_bg, 0, 0, width, 14)

        strength = abs(compound)
        if -0.05 < compound < 0.05:
            strength = 0.35

        fill_width = max(8, int(width * min(strength, 1.0)))
        self.bar_canvas.coords(self.bar_fill, 0, 0, fill_width, 14)
        self.bar_canvas.itemconfig(self.bar_fill, fill=fill_color)

    def _set_result_theme(self, color: str, chip_text: str, title_text: str):
        self.status_chip.set(chip_text)
        self.status_title.set(title_text)
        self.status_chip_label.config(bg=color)
        self.result_card.config(highlightbackground=color)
        self.status_title_label.config(fg=color)

    def analyze_sentiment(self):
        text = self.text_entry.get().strip()

        if not text:
            messagebox.showwarning("Missing text", "Please enter text before analyzing.")
            return

        if self.sia is None:
            messagebox.showerror("Engine unavailable", "Sentiment engine is not available. Please restart the app.")
            return

        score = self.sia.polarity_scores(text)
        compound = score["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
            color = self.SUCCESS
            summary = "This text has a clearly positive emotional tone."
            chip = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
            color = self.ERROR
            summary = "This text has a clearly negative emotional tone."
            chip = "Negative"
        else:
            sentiment = "Neutral"
            color = self.NEUTRAL
            summary = "This text is mostly neutral or balanced in tone."
            chip = "Neutral"

        confidence = max(abs(compound), 1 - score["neu"])

        self.primary_result.set(f"Sentiment: {sentiment}")
        self.metric_sentiment.set(sentiment)
        self.metric_compound.set(f"{compound:.3f}")
        self.metric_confidence.set(f"{confidence:.0%}")
        self.detail_text.set(
            f"{summary}\n"
            f"Positive: {score['pos']:.3f}   Neutral: {score['neu']:.3f}   Negative: {score['neg']:.3f}"
        )

        self._set_result_theme(color=color, chip_text=chip, title_text=f"{sentiment} sentiment detected")
        self._update_strength_bar(compound=compound, fill_color=color)


def main():
    root = tk.Tk()
    PremiumSentimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

