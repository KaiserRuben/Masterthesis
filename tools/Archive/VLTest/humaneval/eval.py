import json
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ===================== config =====================
DATA_PATH = os.path.join(os.path.dirname(__file__), "selected_samples_shuffled.json")
IMAGE_MAX_SIZE = (680, 520)  # max display size (w, h)
IMAGE_PATH = "./images_src"


# ===================== login page =====================
class UsernamePrompt:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluation Login")
        self.root.geometry("400x180")

        tk.Label(root, text="Enter your username:", font=("Arial", 12)).pack(pady=20)
        self.entry = tk.Entry(root, font=("Arial", 12))
        self.entry.pack(pady=5)

        tk.Button(root, text="Start Evaluation", command=self.start).pack(pady=10)
        self.username = None

    def start(self):
        name = self.entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a username before starting.")
            return
        self.username = name
        self.root.destroy()


# ===================== main page =====================
class ImageTextReviewer:
    def __init__(self, root, username):
        self.root = root
        self.username = username
        self.RESULT_PATH = f"{username}_results.json"

        self.root.title(f"Evaluation - User: {username}")
        self.root.geometry("1200x720")

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.index = 0
        self.results_list = []
        self.result_index_by_key = {}

        if os.path.exists(self.RESULT_PATH):
            try:
                with open(self.RESULT_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self.results_list = loaded
            except Exception:
                self.results_list = []

        for i, rec in enumerate(self.results_list):
            key = (rec.get("Index"), rec.get("Task", ""))
            self.result_index_by_key[key] = i

        # ================= UI =================
        self.container = tk.Frame(root, bg="#1e1e1e")
        self.container.pack(fill=tk.BOTH, expand=True)

        # ----- titles -----
        self.label_left = tk.Label(
            self.container,
            text="Original Image & Text",
            bg="#1e1e1e",
            fg="#cccccc",
            font=("Arial", 11, "bold"),
        )
        self.label_left.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=0.04)

        self.label_right = tk.Label(
            self.container,
            text="Mutated Image & Text",
            bg="#1e1e1e",
            fg="#cccccc",
            font=("Arial", 11, "bold"),
        )
        self.label_right.place(relx=0.5, rely=0.0, relwidth=0.5, relheight=0.04)

        # ----- left panel -----
        self.left_panel = tk.Frame(self.container, bg="#272822")
        self.left_panel.place(relx=0, rely=0.04, relwidth=0.5, relheight=0.89)

        self.left_image_label = tk.Label(self.left_panel, bg="#272822")
        self.left_image_label.pack(pady=(10, 5))

        self.left_ref_image_label = tk.Label(self.left_panel, bg="#272822")
        self.left_ref_image_label.pack(pady=(5, 10))

        self.left_text_label = tk.Label(
            self.left_panel,
            text="",
            wraplength=520,
            font=("Arial", 11),
            fg="white",
            bg="#272822",
            justify="center",
        )
        self.left_text_label.pack(padx=10, pady=10)

        # ----- right panel -----
        self.right_panel = tk.Frame(self.container, bg="#272822")
        self.right_panel.place(relx=0.5, rely=0.04, relwidth=0.5, relheight=0.89)

        self.right_image_label = tk.Label(self.right_panel, bg="#272822")
        self.right_image_label.pack(pady=(10, 5))

        self.right_ref_image_label = tk.Label(self.right_panel, bg="#272822")
        self.right_ref_image_label.pack(pady=(5, 10))

        self.right_text_label = tk.Label(
            self.right_panel,
            text="",
            wraplength=520,
            font=("Arial", 11),
            fg="white",
            bg="#272822",
            justify="center",
        )
        self.right_text_label.pack(padx=10, pady=10)

        # ----- bottom rating -----
        bottom = tk.Frame(root, bg="#f5f5f5")
        bottom.place(relx=0, rely=0.93, relwidth=1, relheight=0.07)

        self.progress_label = tk.Label(
            bottom, text="", font=("Arial", 10, "bold"), fg="#0078D7", bg="#f5f5f5"
        )
        self.progress_label.pack(side=tk.LEFT, padx=10)

        tk.Label(
            bottom, text="Img-SemPres:", font=("Arial", 10, "bold"), bg="#f5f5f5"
        ).pack(side=tk.LEFT, padx=(5, 2))
        self.isp_var = tk.IntVar()

        self.isp_labels = []
        for i in range(1, 6):
            lbl = self.make_rating_label(
                bottom, str(i), lambda v=i: self.set_rating("isp", v)
            )
            lbl.pack(side=tk.LEFT, padx=3)
            self.isp_labels.append(lbl)

        tk.Label(
            bottom, text="  Txt-SemPres:", font=("Arial", 10, "bold"), bg="#f5f5f5"
        ).pack(side=tk.LEFT, padx=(10, 2))
        self.tsp_var = tk.IntVar()

        self.tsp_labels = []
        for i in range(1, 6):
            lbl = self.make_rating_label(
                bottom, str(i), lambda v=i: self.set_rating("tsp", v)
            )
            lbl.pack(side=tk.LEFT, padx=3)
            self.tsp_labels.append(lbl)

        tk.Label(
            bottom, text="  ImgTxt-Align:", font=("Arial", 10, "bold"), bg="#f5f5f5"
        ).pack(side=tk.LEFT, padx=(15, 2))
        self.ita_var = tk.IntVar()

        self.ita_labels = []
        for i in range(1, 6):
            lbl = self.make_rating_label(
                bottom, str(i), lambda v=i: self.set_rating("ita", v)
            )
            lbl.pack(side=tk.LEFT, padx=3)
            self.ita_labels.append(lbl)

        tk.Button(bottom, text="Previous", command=self.prev_sample).pack(
            side=tk.RIGHT, padx=6
        )
        tk.Button(bottom, text="Save and Next", command=self.next_sample).pack(
            side=tk.RIGHT, padx=6
        )
        tk.Button(bottom, text="Save", command=self.save_score).pack(
            side=tk.RIGHT, padx=6
        )

        self.feedback_label = tk.Label(
            root, text="", font=("Arial", 9), bg="#222222", fg="white"
        )
        self.feedback_label.place_forget()

        self.show_sample()

    # ================= utils =================
    def make_rating_label(self, parent, text, command):
        label = tk.Label(
            parent,
            text=text,
            font=("Arial", 12, "bold"),
            fg="black",
            bg="#f5f5f5",
            cursor="hand2",
            padx=6,
        )
        label.bind("<Button-1>", lambda e: command())
        label.bind("<Enter>", lambda e: label.config(fg="#ff4d4f"))
        label.bind("<Leave>", lambda e: self.update_colors())
        return label

    def set_rating(self, which, value):
        if which == "isp":
            self.isp_var.set(value)
        elif which == "tsp":
            self.tsp_var.set(value)
        else:
            self.ita_var.set(value)
        self.update_colors()

    def update_colors(self):
        for i, lbl in enumerate(self.isp_labels, start=1):
            lbl.config(
                fg="#d00000" if self.isp_var.get() == i else "black",
                relief="sunken" if self.isp_var.get() == i else "flat",
            )
        for i, lbl in enumerate(self.tsp_labels, start=1):
            lbl.config(
                fg="#d00000" if self.tsp_var.get() == i else "black",
                relief="sunken" if self.tsp_var.get() == i else "flat",
            )
        for i, lbl in enumerate(self.ita_labels, start=1):
            lbl.config(
                fg="#d00000" if self.ita_var.get() == i else "black",
                relief="sunken" if self.ita_var.get() == i else "flat",
            )

    def load_image(self, path, max_size=None):
        if not os.path.exists(path):
            return None

        img = Image.open(path).convert("RGB")
        w, h = img.size

        if max_size is None:
            max_w, max_h = IMAGE_MAX_SIZE
        else:
            max_w, max_h = max_size

        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        img = img.resize((new_w, new_h), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    # ================= core logic =================
    def show_sample(self):
        item = self.data[self.index]

        self.left_img = self.load_image(os.path.join(IMAGE_PATH, item["orig_image"]))
        self.right_img = self.load_image(os.path.join(IMAGE_PATH, item["adv_image"]))

        self.left_image_label.config(image=self.left_img)
        self.right_image_label.config(image=self.right_img)

        self.left_text_label.config(text=item["orig_text"])
        self.right_text_label.config(text=item["adv_text"])

        if item["Task"] == "nlvr" and "ref_image" in item:
            # NLVR: half height
            nlvr_size = (IMAGE_MAX_SIZE[0], IMAGE_MAX_SIZE[1] // 2)

            self.left_img = self.load_image(
                os.path.join(IMAGE_PATH, item["orig_image"]),
                max_size=nlvr_size,
            )
            self.left_ref_img = self.load_image(
                os.path.join(IMAGE_PATH, item["ref_image"]),
                max_size=nlvr_size,
            )

            self.right_img = self.load_image(
                os.path.join(IMAGE_PATH, item["adv_image"]),
                max_size=nlvr_size,
            )
            self.right_ref_img = self.left_ref_img

            self.left_image_label.config(image=self.left_img)
            self.left_ref_image_label.config(image=self.left_ref_img)

            self.right_image_label.config(image=self.right_img)
            self.right_ref_image_label.config(image=self.right_ref_img)

            self.left_ref_image_label.pack(before=self.left_text_label)
            self.right_ref_image_label.pack(before=self.right_text_label)
        else:
            # NOT NLVR: full height
            self.left_img = self.load_image(
                os.path.join(IMAGE_PATH, item["orig_image"])
            )
            self.right_img = self.load_image(
                os.path.join(IMAGE_PATH, item["adv_image"])
            )

            self.left_image_label.config(image=self.left_img)
            self.right_image_label.config(image=self.right_img)

            self.left_ref_image_label.pack_forget()
            self.right_ref_image_label.pack_forget()

        total = len(self.data)
        self.progress_label.config(text=f"Sample {self.index + 1} / {total}")
        self.root.title(
            f"Evaluation - {self.index + 1}/{total} - User: {self.username}"
        )

        self.isp_var.set(0)
        self.tsp_var.set(0)
        self.ita_var.set(0)

        key = (item.get("Index"), item.get("Task", ""))
        idx = self.result_index_by_key.get(key)
        if idx is not None:
            rec = self.results_list[idx]
            self.isp_var.set(rec.get("isp", 0))
            self.tsp_var.set(rec.get("tsp", 0))
            self.ita_var.set(rec.get("ita", 0))
        self.update_colors()

    def save_score(self):
        isp, tsp, ita = (
            self.isp_var.get(),
            self.tsp_var.get(),
            self.ita_var.get(),
        )
        if isp == 0 or tsp == 0 or ita == 0:
            self.show_feedback("Please rate both items before saving")
            return False

        item = self.data[self.index]
        rec = {
            "Index": item.get("Index"),
            "Task": item.get("Task", ""),
            "method": item.get("method", ""),
            "isp": isp,
            "tsp": tsp,
            "ita": ita,
        }

        key = (item.get("Index"), item.get("Task", ""))
        idx = self.result_index_by_key.get(key)

        if idx is None:
            self.results_list.append(rec)
            self.result_index_by_key[key] = len(self.results_list) - 1
        else:
            self.results_list[idx] = rec

        with open(self.RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(self.results_list, f, indent=2, ensure_ascii=False)

        self.show_feedback("✔ Saved")
        return True

    def show_feedback(self, msg):
        self.feedback_label.config(text=msg)
        self.feedback_label.place(relx=0.95, rely=0.95, anchor="se")
        self.root.after(1200, self.feedback_label.place_forget)

    def next_sample(self):
        if not self.save_score():
            return
        if self.index < len(self.data) - 1:
            self.index += 1
            self.show_sample()

    def prev_sample(self):
        if self.index > 0:
            self.index -= 1
            self.show_sample()


# ===================== entry =====================
if __name__ == "__main__":
    temp_root = tk.Tk()
    prompt = UsernamePrompt(temp_root)
    temp_root.mainloop()

    if prompt.username:
        root = tk.Tk()
        app = ImageTextReviewer(root, prompt.username)
        root.mainloop()
