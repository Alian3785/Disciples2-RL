# data_dicts_battle_selector_gui.py
# Optional Tkinter UI for picking Red/Blue units. Run only as __main__:
#   python data_dicts_battle_selector_gui.py
# (Keeps tkinter off the import path of data_dicts_compact_lines for headless/CI.)
import os
import subprocess
import sys
from pprint import pformat

from tkinter import E, W, Button, Frame, Label, StringVar, Tk, Toplevel, messagebox
from tkinter.ttk import Combobox

from data_dicts_compact_lines import (
    DATA,
    NO_SELECTION,
    build_name_index,
    first_by_name,
    map_unit_to_battle,
    placeholder_unit,
    write_units_to_file,
)


def run_battle(py_path: str) -> None:
    try:
        subprocess.Popen([sys.executable, py_path])
    except Exception as e:
        messagebox.showerror("Ошибка запуска", f"Не удалось запустить {py_path}\n{e}")


class BattleSelectorGUI:
    def __init__(self, master, data_list):
        self.master = master
        self.data_list = data_list
        self.name_index = build_name_index(data_list)

        master.title("Red vs Blue — выбор юнитов (передняя/задняя линии)")
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=0)
        master.grid_columnconfigure(2, weight=1)

        self.names_all = [NO_SELECTION] + list(self.name_index.keys())

        Label(master, text="RED").grid(row=0, column=0, pady=(10, 5))
        Label(master, text=" ").grid(row=0, column=1)
        Label(master, text="BLUE").grid(row=0, column=2, pady=(10, 5))

        Label(master, text="Передняя линия (pos 1–3)").grid(row=1, column=0, pady=(0, 4))
        Label(master, text=" ").grid(row=1, column=1)
        Label(master, text="Передняя линия (pos 7–9)").grid(row=1, column=2, pady=(0, 4))

        self.red_ahead_vars = [StringVar(value=NO_SELECTION) for _ in range(3)]
        self.blue_ahead_vars = [StringVar(value=NO_SELECTION) for _ in range(3)]
        self.red_ahead_boxes = []
        self.blue_ahead_boxes = []

        for i in range(3):
            cb = Combobox(
                master,
                textvariable=self.red_ahead_vars[i],
                values=self.names_all,
                width=36,
                state="readonly",
            )
            cb.grid(row=2 + i, column=0, padx=10, pady=3, sticky=E + W)
            self.red_ahead_boxes.append(cb)

            cb2 = Combobox(
                master,
                textvariable=self.blue_ahead_vars[i],
                values=self.names_all,
                width=36,
                state="readonly",
            )
            cb2.grid(row=2 + i, column=2, padx=10, pady=3, sticky=E + W)
            self.blue_ahead_boxes.append(cb2)

        Label(master, text="Задняя линия (pos 4–6)").grid(row=5, column=0, pady=(8, 4))
        Label(master, text=" ").grid(row=5, column=1)
        Label(master, text="Задняя линия (pos 10–12)").grid(row=5, column=2, pady=(8, 4))

        self.red_behind_vars = [StringVar(value=NO_SELECTION) for _ in range(3)]
        self.blue_behind_vars = [StringVar(value=NO_SELECTION) for _ in range(3)]
        self.red_behind_boxes = []
        self.blue_behind_boxes = []

        for i in range(3):
            cb = Combobox(
                master,
                textvariable=self.red_behind_vars[i],
                values=self.names_all,
                width=36,
                state="readonly",
            )
            cb.grid(row=6 + i, column=0, padx=10, pady=3, sticky=E + W)
            self.red_behind_boxes.append(cb)

            cb2 = Combobox(
                master,
                textvariable=self.blue_behind_vars[i],
                values=self.names_all,
                width=36,
                state="readonly",
            )
            cb2.grid(row=6 + i, column=2, padx=10, pady=3, sticky=E + W)
            self.blue_behind_boxes.append(cb2)

        btn_frame = Frame(master)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=(12, 10))
        Button(btn_frame, text="Очистить", command=self.clear_all).grid(row=0, column=0, padx=5)
        Button(btn_frame, text="Предпросмотр JSON", command=self.preview).grid(
            row=0, column=1, padx=5
        )
        Button(btn_frame, text="Записать в 1workingbattle.py", command=self.save_to_file).grid(
            row=0, column=2, padx=5
        )
        Button(
            btn_frame, text="Записать и запустить бой", command=self.save_and_run
        ).grid(row=0, column=3, padx=5)

        self._all_vars = (
            self.red_ahead_vars
            + self.red_behind_vars
            + self.blue_ahead_vars
            + self.blue_behind_vars
        )

    def _u_by_name(self, name):
        if not name or name == NO_SELECTION:
            return None
        return first_by_name(name, self.name_index)

    def _collect_units_compact(self):
        red_positions = {}
        blue_positions = {}

        for idx, v in enumerate(self.red_ahead_vars):
            name = v.get()
            pos = 1 + idx
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    red_positions[pos] = map_unit_to_battle(u, "red", pos)

        for idx, v in enumerate(self.red_behind_vars):
            name = v.get()
            pos = 4 + idx
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    red_positions[pos] = map_unit_to_battle(u, "red", pos)

        for idx, v in enumerate(self.blue_ahead_vars):
            name = v.get()
            pos = 7 + idx
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    blue_positions[pos] = map_unit_to_battle(u, "blue", pos)

        for idx, v in enumerate(self.blue_behind_vars):
            name = v.get()
            pos = 10 + idx
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    blue_positions[pos] = map_unit_to_battle(u, "blue", pos)

        if (not red_positions) or (not blue_positions):
            messagebox.showwarning(
                "Недостаточно юнитов",
                "Нужно выбрать минимум по одному юниту на каждую сторону.",
            )
            return None, None

        units_red = [red_positions[p] for p in range(1, 7) if p in red_positions]
        units_blue = [blue_positions[p] for p in range(7, 13) if p in blue_positions]
        return units_red, units_blue

    def _collect_units_with_placeholders(self):
        red_positions = {i: None for i in range(1, 7)}
        blue_positions = {i: None for i in range(7, 13)}

        for idx, v in enumerate(self.red_ahead_vars):
            pos = 1 + idx
            name = v.get()
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    red_positions[pos] = map_unit_to_battle(u, "red", pos)
        for idx, v in enumerate(self.red_behind_vars):
            pos = 4 + idx
            name = v.get()
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    red_positions[pos] = map_unit_to_battle(u, "red", pos)

        for idx, v in enumerate(self.blue_ahead_vars):
            pos = 7 + idx
            name = v.get()
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    blue_positions[pos] = map_unit_to_battle(u, "blue", pos)
        for idx, v in enumerate(self.blue_behind_vars):
            pos = 10 + idx
            name = v.get()
            if name != NO_SELECTION:
                u = self._u_by_name(name)
                if u:
                    blue_positions[pos] = map_unit_to_battle(u, "blue", pos)

        any_red = any(red_positions[p] is not None for p in red_positions)
        any_blue = any(blue_positions[p] is not None for p in blue_positions)
        if not any_red or not any_blue:
            messagebox.showwarning(
                "Недостаточно юнитов",
                "Нужно выбрать минимум по одному юниту на каждую сторону.",
            )
            return None, None

        units_red = [
            red_positions[p] if red_positions[p] is not None else placeholder_unit("red", p)
            for p in range(1, 7)
        ]
        units_blue = [
            blue_positions[p] if blue_positions[p] is not None else placeholder_unit("blue", p)
            for p in range(7, 13)
        ]
        return units_red, units_blue

    def preview(self):
        from tkinter.scrolledtext import ScrolledText

        units_red, units_blue = self._collect_units_with_placeholders()
        if units_red is None:
            return
        msg = (
            "UNITS_RED = "
            + pformat(units_red, width=120, compact=False, sort_dicts=False)
            + "\n\nUNITS_BLUE = "
            + pformat(units_blue, width=120, compact=False, sort_dicts=False)
        )
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(msg)
            self.master.update()
        except Exception as e:
            messagebox.showwarning("Буфер обмена", f"Не удалось скопировать в буфер обмена:\n{e}")

        top = Toplevel(self.master)
        top.title("Предпросмотр (скопировано в буфер обмена)")
        st = ScrolledText(top, width=140, height=44)
        st.pack(fill="both", expand=True)
        st.insert("1.0", msg)
        st.configure(state="disabled")
        Label(top, text="Содержимое предпросмотра скопировано в буфер обмена.").pack(pady=(4, 0))
        Button(top, text="Закрыть", command=top.destroy).pack(pady=6)

    def save_to_file(self):
        units_red, units_blue = self._collect_units_compact()
        if units_red is None:
            return
        py_path = os.path.join(os.getcwd(), "1workingbattle.py")
        try:
            write_units_to_file(py_path, units_red, units_blue)
        except Exception as e:
            messagebox.showerror("Ошибка записи", f"Не удалось записать в {py_path}\n{e}")
            return
        messagebox.showinfo(
            "Готово",
            f"UNITS_RED/UNITS_BLUE записаны в {py_path} (кастомный блок в конце файла).",
        )

    def save_and_run(self):
        self.save_to_file()
        py_path = os.path.join(os.getcwd(), "1workingbattle.py")
        if os.path.exists(py_path):
            run_battle(py_path)

    def clear_all(self):
        for v in self._all_vars:
            v.set(NO_SELECTION)


if __name__ == "__main__":
    try:
        DATA  # noqa: B018
    except NameError:
        raise SystemExit(
            "Ошибка: переменная DATA не найдена. Импортируй/объяви DATA перед запуском GUI."
        )

    root = Tk()
    BattleSelectorGUI(root, DATA)
    root.mainloop()
