import pickle
import os.path
import time

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog, messagebox

import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
import re

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



class DrawingClassifier:

    def __init__(self):
        self.class0, self.class1, self.class2, self.class3, self.class4, self.class5, self.class6, self.class7, \
            self.class8, self.class9 = None, None, None, None, None, None, None, None, None, None
        self.class10, self.class11, self.class12, self.class13 = None, None, None, None
        # +, -, *, u (qui est la variable)
        self.class0_counter, self.class1_counter, self.class2_counter, self.class3_counter, self.class4_counter, \
            self.class5_counter, self.class6_counter, self.class7_counter, self.class8_counter, \
            self.class9_counter = None, None, None, None, None, None, None, None, None, None
        self.class10_counter, self.class11_counter, self.class12_counter, self.class13_counter = None, None, None, None
        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None
        self.image_digit = None

        self.status_label = None
        self.canvas = None
        self.draw = None

        self.end_result = 0
        self.train_ = False

        self.brush_width = 15

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name down below!",
                                                parent=msg)
        if os.path.exists(self.proj_name):
            self.train_ = True
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class0 = data['c0']
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            self.class4 = data['c4']
            self.class5 = data['c5']
            self.class6 = data['c6']
            self.class7 = data['c7']
            self.class8 = data['c8']
            self.class9 = data['c9']
            self.class10 = data['c10']
            self.class11 = data['c11']
            self.class12 = data['c12']
            self.class13 = data['c13']
            self.class0_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class0}") if fichier.endswith(".png"))
            self.class1_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class1}") if fichier.endswith(".png"))
            self.class2_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class2}") if fichier.endswith(".png"))
            self.class3_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class3}") if fichier.endswith(".png"))
            self.class4_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class4}") if fichier.endswith(".png"))
            self.class5_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class5}") if fichier.endswith(".png"))
            self.class6_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class6}") if fichier.endswith(".png"))
            self.class7_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class7}") if fichier.endswith(".png"))
            self.class8_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class8}") if fichier.endswith(".png"))
            self.class9_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class9}") if fichier.endswith(".png"))
            self.class10_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class10}") if fichier.endswith(".png"))
            self.class11_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class11}") if fichier.endswith(".png"))
            self.class12_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class12}") if fichier.endswith(".png"))
            self.class13_counter = sum(
                1 for fichier in os.listdir(f"{self.proj_name}/{self.class13}") if fichier.endswith(".png"))
            self.clf = data['clf']
            self.proj_name = data['pname']
        else:
            self.class0 = "0"
            self.class1 = "1"
            self.class2 = "2"
            self.class3 = "3"
            self.class4 = "4"
            self.class5 = "5"
            self.class6 = "6"
            self.class7 = "7"
            self.class8 = "8"
            self.class9 = "9"
            self.class10 = "10"  # +
            self.class11 = "11"  # -
            self.class12 = "12"  # *
            self.class13 = "13"  # u

            self.class0_counter = 87
            self.class1_counter = 70
            self.class2_counter = 71
            self.class3_counter = 71
            self.class4_counter = 71
            self.class5_counter = 71
            self.class6_counter = 71
            self.class7_counter = 86
            self.class8_counter = 71
            self.class9_counter = 71
            self.class10_counter = 21
            self.class11_counter = 12
            self.class12_counter = 25
            self.class13_counter = 11

            self.clf = LinearSVC()

            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class0)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.mkdir(self.class4)
            os.mkdir(self.class5)
            os.mkdir(self.class6)
            os.mkdir(self.class7)
            os.mkdir(self.class8)
            os.mkdir(self.class9)
            os.mkdir(self.class10)
            os.mkdir(self.class11)
            os.mkdir(self.class12)
            os.mkdir(self.class13)
            os.chdir("..")

    def init_gui(self):

        WIDTH = 1000
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"NeuralNine Drawing Classifier Alpha v0.2 - {self.proj_name}")

        self.canvas = Canvas(self.root, width=WIDTH - 10, height=HEIGHT - 10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        if not self.train_:
            btn_frame.columnconfigure(3, weight=1)

            class0_btn = Button(btn_frame, text=self.class0, command=lambda: self.save(0))
            class0_btn.grid(row=0, column=0, sticky=W + E)

            class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
            class1_btn.grid(row=0, column=1, sticky=W + E)

            class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
            class2_btn.grid(row=0, column=2, sticky=W + E)

            class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
            class3_btn.grid(row=0, column=3, sticky=W + E)

            class4_btn = Button(btn_frame, text=self.class4, command=lambda: self.save(4))
            class4_btn.grid(row=1, column=0, sticky=W + E)

            class5_btn = Button(btn_frame, text=self.class5, command=lambda: self.save(5))
            class5_btn.grid(row=1, column=1, sticky=W + E)

            class6_btn = Button(btn_frame, text=self.class6, command=lambda: self.save(6))
            class6_btn.grid(row=1, column=2, sticky=W + E)

            class7_btn = Button(btn_frame, text=self.class7, command=lambda: self.save(7))
            class7_btn.grid(row=1, column=3, sticky=W + E)

            clear_btn = Button(btn_frame, text="Clear", command=self.clear)
            clear_btn.grid(row=2, column=0, sticky=W + E)

            class8_btn = Button(btn_frame, text=self.class8, command=lambda: self.save(8))
            class8_btn.grid(row=2, column=1, sticky=W + E)

            class9_btn = Button(btn_frame, text=self.class9, command=lambda: self.save(9))
            class9_btn.grid(row=2, column=2, sticky=W + E)

            class10_btn = Button(btn_frame, text=" + ", command=lambda: self.save(10))
            class10_btn.grid(row=2, column=3, sticky=W + E)

            save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
            save_btn.grid(row=3, column=0, sticky=W + E)

            save_everything_btn = Button(btn_frame, text="Save Everything", command=self.save_everything)
            save_everything_btn.grid(row=3, column=1, sticky=W + E)

            change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
            change_btn.grid(row=3, column=2, sticky=W + E)

            train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
            train_btn.grid(row=3, column=3, sticky=W + E)

            class11_btn = Button(btn_frame, text= " - ", command=lambda: self.save(11))
            class11_btn.grid(row=4, column=0, sticky=W + E)

            class12_btn = Button(btn_frame, text=" * ", command=lambda: self.save(12))
            class12_btn.grid(row=4, column=1, sticky=W + E)

            class13_btn = Button(btn_frame, text=" u ", command=lambda: self.save(13))
            class13_btn.grid(row=4, column=2, sticky=W + E)
        else:

            clear_btn = Button(btn_frame, text="Clear", command=self.clear)
            clear_btn.grid(row=0, column=1, sticky=W + E)

            change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
            change_btn.grid(row=0, column=0, sticky=W + E)

            train_model_btn = Button(btn_frame, text="Train Model", command=self.train_model)
            train_model_btn.grid(row=0, column=2, sticky=W + E)

            predict_btn = Button(btn_frame, text="Predict", command=self.predict_number)
            predict_btn.grid(row=1, column=1, sticky=W + E)


        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=5, column=1, sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black",
                            width=self.brush_width)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def save(self, class_num):
        self.image1.save("temp.png")  # Sauvegarder l'image temporairement
        img = PIL.Image.open("temp.png").convert("L")  # Charger et convertir en niveaux de gris

        # Segmenter l'image en chiffres
        digits = self.segment_image(img)

        if not digits:
            print("Aucun chiffre détecté.")
            return

        # Sauvegarder chaque chiffre détecté
        for i, digit_array in enumerate(digits):
            # Reconstruire l'image d'origine avec uniquement le chiffre détecté
            digit_image = PIL.Image.fromarray(digit_array.reshape(50, 50)).convert("L")

            # Inverser les couleurs (blanc <-> noir)
            digit_image = PIL.ImageOps.invert(digit_image)

            # Choisir le dossier et le compteur en fonction de la classe
            folder = f"{self.proj_name}/{getattr(self, f'class{class_num}')}"
            counter_attr = f'class{class_num}_counter'
            counter = getattr(self, counter_attr)
            setattr(self, counter_attr, counter + 1)

            # Sauvegarder l'image du chiffre dans le bon dossier
            digit_image.save(f"{folder}/{counter}.png", "PNG")

        self.clear()  # Réinitialiser après sauvegarde

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()

    def save_everything(self):
        data = {"c0": self.class0, "c1": self.class1, "c2": self.class2, "c3": self.class3, "c4": self.class4,
                "c5": self.class5, "c6": self.class6, "c7": self.class7, "c8": self.class8, "c9": self.class9,
                "c10": self.class10, "c11": self.class11, "c12": self.class12, "c13": self.class13,
                "c0c": self.class0_counter, "c1c": self.class1_counter, "c2c": self.class2_counter,
                "c3c": self.class3_counter, "c4c": self.class4_counter, "c5c": self.class5_counter,
                "c6c": self.class6_counter, "c7c": self.class7_counter, "c8c": self.class8_counter,
                "c9c": self.class9_counter, "c10c": self.class10_counter, "c11c": self.class11_counter,
                "c12c": self.class12_counter, "c13c": self.class13_counter,
                "clf": self.clf, "pname": self.proj_name}
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Project successfully saved!", parent=self.root)

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])
        # convertit en vecteur les images pour les préparer

        for class_num, class_counter in enumerate([
            self.class0_counter, self.class1_counter, self.class2_counter, self.class3_counter,
            self.class4_counter, self.class5_counter, self.class6_counter, self.class7_counter,
            self.class8_counter, self.class9_counter, self.class10_counter, self.class11_counter,
            self.class12_counter, self.class13_counter]):

            for x in range(1, class_counter):
                img = cv.imread(f"{self.proj_name}/{getattr(self, f'class{class_num}')}/{x}.png")[:, :, 0]
                img = img.reshape(2500)
                img_list = np.append(img_list, [img])
                class_list = np.append(class_list, class_num)

        img_list = img_list.reshape(self.class0_counter - 1 + self.class1_counter - 1 + self.class2_counter - 1
                                    + self.class3_counter - 1 + self.class4_counter - 1 + self.class5_counter - 1
                                    + self.class6_counter - 1 + self.class7_counter - 1 + self.class8_counter - 1
                                    + self.class9_counter - 1 + self.class10_counter - 1 + self.class11_counter - 1
                                    + self.class12_counter - 1 + self.class13_counter - 1, 2500)
        # transforme en une matrice de nombre d'eléménts appartenant à la classe fois le nombre de vecteur par image
        self.clf.fit(img_list, class_list)
        # entrainement du model
        # il a apprit à associer un vecteur d'image à une étiquette
        tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Model successfully trained!", parent=self.root)
        # pop up d'information
        self.train_ = True

    def predict(self, i):

        self.image1.save("temp.png")
        # sauvegarde image
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)
        # sauvegarde image redimansionnée

        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        # conversion de l'image en tableau
        prediction = self.clf.predict([img])
        # la suite et les message de prédiciton

        # self.clear()
        print(f"Prediction: {prediction[0]}")
        return prediction[0]

    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()

        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Model successfully saved!", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo("NeuralNine Drawing Classifier", "Model successfully loaded!", parent=self.root)

    def predict_number(self):
        # Sauvegarder et charger l'image dessinée
        self.image1.save("temp_second.png")
        img = PIL.Image.open("temp_second.png").convert("L")

        number = ""

        # Segmenter l'image en chiffres
        digits = self.segment_image(img)
        if not digits:
            tkinter.messagebox.showinfo("Résultat", "Aucun chiffre détecté.")
            return

        # Sauvegarder et prédire chaque chiffre
        for i, digit_array in enumerate(digits):
            # Reconstruire l'image du chiffre détecté
            digit_image = PIL.Image.fromarray(digit_array.reshape(50, 50)).convert("L")

            # Inverser les couleurs (blanc <-> noir) pour correspondre au modèle attendu
            digit_image = PIL.ImageOps.invert(digit_image)

            # Sauvegarder l'image du chiffre
            digit_image.save(f"digit_{i + 1}_isolated.png", "PNG")
            img1 = PIL.Image.open(f"digit_{i + 1}_isolated.png")
            img1.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)
            img1.save("predictshape.png", "PNG")
            img1 = cv.imread("predictshape.png")[:, :, 0]
            img1 = img1.reshape(2500)
            predicted_digit = self.clf.predict([img1])

            # de base c'était ça
            # predicted_digit = self.predict(i)
            if predicted_digit[0] == 10:
                number += "+"
            elif predicted_digit[0] == 11:
                number += "-"
            elif predicted_digit[0] == 12:
                number += "*"
            elif predicted_digit[0] == 13:
                number += "u"
            else:
                number += str(int(predicted_digit[0]))

            self.clear()
            self.root.attributes("-topmost", True)

        # Afficher le résultat
        if "u" in number:
            # Extract coefficients from the string
            match = re.match(r"(\+|-)?(\d*)u(\+|-)?(\d+)?", number)
            if match:
                a_sign = match.group(1) if match.group(1) else "+"
                a = int(match.group(2)) if match.group(2) else 1
                b_sign = match.group(3) if match.group(3) else "+"
                b = int(match.group(4)) if match.group(4) else 0

                # Define the function
                def f(u):
                    a_val = a if a_sign == "+" else -a
                    b_val = b if b_sign == "+" else -b
                    return a_val * u + b_val

                # Generate values for u
                u_values = np.linspace(-10, 10, 400)
                f_values = f(u_values)

                # Plot the function
                plt.plot(u_values, f_values, label=f"f(u)={a_sign}{a}u{b_sign}{b}")
                plt.xlabel("u")
                plt.ylabel("f(u)")
                plt.title("Graph of the function f(u)")
                plt.legend()
                plt.grid(True)
                plt.show()

            tkinter.messagebox.showinfo("Résultat", f"Nombre prédit : {number}", parent=self.root)
            return
        elif any(op in number for op in ["+", "-", "*"]):
            result = eval(number.replace("u", "*"))
            tkinter.messagebox.showinfo("Résultat", f"Nombre prédit : {number} = {result}", parent=self.root)
            return
        else:
            tkinter.messagebox.showinfo("Résultat", f"Nombre prédit : {number}", parent=self.root)
            return

    def segment_image(self, img):
        # Convertir l'image en numpy array
        img_array = np.array(img)

        # Binarisation (thresholding)
        _, thresh = cv.threshold(img_array, 128, 255, cv.THRESH_BINARY_INV)

        # Détection des contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        digit_contours = []
        for contour in contours:
            # Obtenir la boîte englobante de chaque contour
            x, y, w, h = cv.boundingRect(contour)

            # Filtrer les zones trop petites (éviter les bruits)
            if w > 10 and h > 10:  # Ajuster ces seuils selon vos données
                digit = thresh[y:y + h, x:x + w]

                # Redimensionner à 50x50
                digit_resized = cv.resize(digit, (50, 50), interpolation=cv.INTER_AREA)

                # Ajouter les informations associées (coordonnées x pour trier)
                digit_contours.append((x, digit_resized.flatten()))

        # Trier les chiffres par leur position horizontale (x)
        digit_contours.sort(key=lambda item: item[0])

        # Extraire uniquement les chiffres triés
        digits = [item[1] for item in digit_contours]
        return digits


DrawingClassifier()
