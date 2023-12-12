import base64
import math
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from collections import Counter

import PIL
import cv2
import numpy as np
import pywt
from PIL import Image, ImageTk
from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import io

from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

root = tk.Tk()
root.geometry('1150x700')
root.title('Tkinter Hup')

def home_page():
    def polynomial(LHS, RHS, n):
        for i in range(0, n):
            LHS[0].append(i)
            RHS[0].append(i)
            LHS[1].append((i * i * i) % n)
            RHS[1].append((i * i) % n)

    def points_generate(arr_x, arr_y, n, LHS, RHS):
        count = 0
        print("LHS[1]:", LHS[1])
        print("RHS[1]:", RHS[1])
        for i in range(0, n):
            for j in range(0, n):
                if LHS[1][i] == RHS[1][j]:
                    count += 1
                    arr_x.append(LHS[0][i])
                    arr_y.append(RHS[0][j])

        return count

    def text_to_numeric(text):
        return int.from_bytes(text.encode('utf-8'), 'big')

    def numeric_to_text(number):
        return number.to_bytes((number.bit_length() + 7) // 8, 'big').decode('utf-8')

    def encrypt_decrypt():
        try:
            # Retrieve values from entry widgets
            n = int(entry_n.get())
            d = int(entry_d.get())
            k = int(entry_k.get())
            message_text = entry_message.get()

            # Polynomial
            LHS = [[]]
            RHS = [[]]
            LHS.append([])
            RHS.append([])
            polynomial(LHS, RHS, n)

            arr_x = []
            arr_y = []
            # Generating base points
            count = points_generate(arr_x, arr_y, n, LHS, RHS)

            # Calculation of Base Point
            bx = arr_x[0] if arr_x else 0  # Check if arr_x is not empty
            by = arr_y[0] if arr_y else 0  # Check if arr_y is not empty

            # Q i.e. sender's public key generation
            Qx = d * bx
            Qy = d * by

            # Encryption
            M = text_to_numeric(message_text)

            # Cipher text 1 generation
            C1x = k * Qx
            C1y = k * Qy

            # Cipher text 2 generation
            C2x = k * Qx + M
            C2y = k * Qy + M

            # Decryption
            Mx = C2x - d * C1x
            My = C2y - d * C1y
            decrypted_message = numeric_to_text(Mx)

            # Display the results in the text widget
            result_text = f"Encrypted Message: ({C1x}, {C1y}), ({C2x}, {C2y})\nDecrypted Message: {decrypted_message}"
            result_display.delete(1.0, tk.END)
            result_display.insert(tk.END, result_text)

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numeric values.")

    home_frame = tk.Frame(main_frame)

    label_n = tk.Label(main_frame, text="Enter the prime number 'P':")
    label_n.pack(pady=10)

    entry_n = tk.Entry(main_frame)
    entry_n.pack(pady=10)

    label_d = tk.Label(main_frame, text="Enter the random number 'x' (Private key of Sender):")
    label_d.pack(pady=10)

    entry_d = tk.Entry(main_frame)
    entry_d.pack(pady=10)

    label_k = tk.Label(main_frame, text="Enter the random number 'g' (Encryption key):")
    label_k.pack(pady=10)

    entry_k = tk.Entry(main_frame)
    entry_k.pack(pady=10)

    label_message = tk.Label(main_frame, text="Enter the message to be sent:")
    label_message.pack(pady=10)

    entry_message = tk.Entry(main_frame)
    entry_message.pack(pady=10)

    encrypt_button = tk.Button(main_frame, text="Encrypt/Decrypt", command=encrypt_decrypt)
    encrypt_button.pack(pady=10)

    result_display = tk.Text(main_frame, height=10, width=50)
    result_display.pack(pady=10)
    home_frame.pack(pady=20)



def menu_page():


    def display_image(image, label):
        # Convertir BGR en RGB pour l'affichage dans Tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Redimensionner l'image pour un meilleur affichage dans l'interface (ajuster si nécessaire)
        image_rgb = cv2.resize(image_rgb, (200, 200))
        # Convertir en format PhotoImage
        photo_image = Image.fromarray(image_rgb)
        photo_image = ImageTk.PhotoImage(photo_image)

        # Mettre à jour l'étiquette avec la nouvelle image
        label.config(image=photo_image)
        label.image = photo_image  # Conserver une référence pour éviter la collecte des déchets

    def choose_image():
        global image_path  # Add this line to modify the global variable
        file_path = filedialog.askopenfilename(title="Sélectionner une Image",
                                               filetypes=[("Fichiers image", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image_path = file_path

            # Afficher l'image originale
            original_image = cv2.imread(image_path)
            display_image(original_image, original_image_label)

            # Réinitialiser les étiquettes de performance
            compressed_image_label.config(image=None)


    def compress_image():

        global image_path,compressed_image_path

        if image_path:
            # Charger l'image en couleur
            original_image = cv2.imread(image_path)

            # Appliquer la transformation DWT à chaque canal de couleur
            compressed_image = np.zeros_like(original_image, dtype=np.float32)
            for i in range(3):  # Pour les canaux Rouge, Vert et Bleu
                channel = original_image[:, :, i]

                # Appliquer la transformation DWT
                coeffs = pywt.dwt2(channel, 'bior1.3')
                cA, (cH, cV, cD) = coeffs

                # Quantification - Ajuster la valeur pour le niveau de compression souhaité
                threshold = 0.0001

                cA, cH, cV, cD = (pywt.threshold(c, threshold, mode='soft') for c in (cA, cH, cV, cD))

                # Reconstruire le canal
                compressed_channel = pywt.idwt2((cA, (cH, cV, cD)), 'bior1.3')

                if compressed_channel.shape == channel.shape:
                    compressed_image[:, :, i] = compressed_channel.astype(np.float32)
                else:
                    # Handle mismatching shapes appropriately (resize or other operation)
                    compressed_channel = cv2.resize(compressed_channel, (channel.shape[1], channel.shape[0]))

                    # Assign the resized compressed channel to the compressed image
                    compressed_image[:, :, i] = compressed_channel.astype(np.float32)

            # Convertir les valeurs à l'intervalle [0, 255]
            compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

            # Afficher l'image compressée
            display_image(compressed_image, compressed_image_label)

            # Enregistrer le chemin de l'image compressée
            compressed_image_path = filedialog.asksaveasfilename(defaultextension=".jpg",filetypes=[("Fichiers JPEG", "*.jpg;*.jpeg")],title="Enregistrer l'Image Compressée")

            # Sauvegarder l'image compressée avec compression JPEG
            quality = 70  # Ajuster la qualité selon vos préférences (0-100)
            success = cv2.imwrite(compressed_image_path, compressed_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

            if success:
                messagebox.showinfo("Succès", f"Image compressée enregistrée à : {compressed_image_path}")
            else:
                messagebox.showerror("Erreur", "Impossible d'enregistrer l'image compressée.")
        else:
            messagebox.showerror("Erreur", "Aucune image sélectionnée. Veuillez choisir une image.")


    def evaluate_performance():

        global image_path,compressed_image_path

        if image_path is None or compressed_image_path is None:
            print("Error: Unable to evaluate performance. Make sure the images are loaded and compressed.")
            return
        original_imageee = cv2.imread(image_path)
        compressed_imageee = cv2.imread(compressed_image_path)


        # Convert images to grayscale
        original_gray = cv2.cvtColor(original_imageee, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed_imageee, cv2.COLOR_BGR2GRAY)

        # Calculate performance metrics
        mse = np.mean((original_imageee - compressed_imageee) ** 2)

        # Calculer le PSNR (Peak Signal-to-Noise Ratio)
        psnr = 20 * math.log10(255 / math.sqrt(mse))

        ssim_value, _ = ssim(original_gray, compressed_gray, full=True)


        mse_label.config(text=f'MSE: {mse}')
        psnr_label.config(text=f'PSNR: {psnr} dB')
        ssim_label.config(text=f'SSIM: {ssim_value}')

        # Ensure that the console output is not blocked
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def crypte_image():
        global image_path,compressed_image_path,encrypted_data

        # Génération de la clé ECC
        private_key = ec.generate_private_key(ec.SECP256R1())

        # Récupération de la clé publique
        public_key = private_key.public_key()

        # Sérialisation de la clé publique
        serialized_public = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Récupération de l'image
        image = Image.open(compressed_image_path)  # Remplace 'ton_image.jpg' par le chemin de ton image
        width, height = image.size
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format=image.format)
        img_bytes = img_byte_array.getvalue()

        # Chiffrement de l'image avec AES
        key = ec.generate_private_key(ec.SECP256R1())
        shared_key = private_key.exchange(ec.ECDH(), key.public_key())
        aes_key = shared_key[:32]  # Utilisation des 32 premiers octets comme clé AES

        # Padding des données de l'image
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(img_bytes) + padder.finalize()

        # Initialisation du chiffrement AES en mode CBC
        iv = os.urandom(16)
        # Remplace par un vecteur d'initialisation approprié
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()

        # Chiffrement des données
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        # Sauvegarde de la clé publique et de l'image chiffrée
        with open('cle_publique.pem', 'wb') as file:
            file.write(serialized_public)

        with open('image_chiffree.enc', 'wb') as file:
            file.write(encrypted_data)

        # Encode the encrypted data to base64 for visualization
        encoded_data = base64.b64encode(encrypted_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)

        # Convert the encoded data to an image
        image_from_encrypted_data = Image.frombytes('RGB', (100, 100), encoded_data)
        image_from_encrypted_data = image_from_encrypted_data.resize((width, height))

        # Save the image representing the encrypted data
        image_from_encrypted_data.save(compressed_image_path + '_encrypted_image.png')


        messagebox.showinfo("Succès", f"Image Chiffré est bien enregistrée ")

        II = cv2.imread('Cryptographie/image_chiffree.jpg')
        display_image(II, Crpted_image_label)

    def histogram():

        global image_path,encrypted_data
        img_original = cv2.imread(image_path, 0)  # Charger l'image en niveaux de gris
        hist_original = Counter(img_original.flatten())

        # Histogramme des données chiffrées
        histogram_encrypted = Counter(encrypted_data)

        # Création du graphique
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(hist_original.keys(), hist_original.values())
        plt.xlabel('Valeur des pixels')
        plt.ylabel('Fréquence')
        plt.title('Histogramme de l\'image originale')

        plt.subplot(1, 2, 2)
        plt.bar(histogram_encrypted.keys(), histogram_encrypted.values())
        plt.xlabel('Valeur des octets')
        plt.ylabel('Fréquence')
        plt.title('Histogramme des octets chiffrés')

        plt.tight_layout()
        plt.show()

    def calculate_correlation(data):
        correlation = sum(data[i] == data[i + 1] for i in range(len(data) - 1)) / (len(data) - 1)
        return correlation

    def dig_correlation():

        global image_path,encrypted_data

        # Chargement de l'image originale
        original_image = Image.open(image_path)
        original_gray = original_image.convert('L')  # Conversion en niveaux de gris
        original_array = np.array(original_gray)

        # Calcul de la corrélation pour l'image originale
        correlation_original = np.corrcoef(original_array[0:-1, 0:-1].flatten(), original_array[1:, 1:].flatten())[0, 1]

        # Conversion de l'image chiffrée en niveaux de gris

        correlation = calculate_correlation(encrypted_data)
        encoded_data = base64.b64encode(encrypted_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        image_size = (100, 100)  # Mettez la taille de l'image ici
        encrypted_image = Image.frombytes('RGB', image_size, encoded_data)

        # Convertir l'image en niveaux de gris pour simplifier la corrélation
        encrypted_gray = encrypted_image.convert('L')
        encrypted_array = np.array(encrypted_gray)

        # Affichage des diagrammes de dispersion
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(original_array[0:-1, 0:-1].flatten(), original_array[1:, 1:].flatten(), s=1, alpha=0.5)
        plt.title(f'Diagramme de corrélation - Image originale\nCorrélation : {correlation_original:.4f}')
        plt.xlabel('Pixel (i, j)')
        plt.ylabel('Pixel (i+1, j+1)')

        plt.subplot(1, 2, 2)
        plt.scatter(encrypted_array[0:-1, 0:-1].flatten(), encrypted_array[1:, 1:].flatten(), s=1, alpha=0.5)
        plt.title(f'Diagramme de corrélation\nCorrélation : {correlation:.4f}')
        plt.xlabel('Pixel (i, j)')
        plt.ylabel('Pixel (i+1, j+1)')

        plt.tight_layout()
        plt.show()

    def calculate_entropy():
        global image_path,encrypted_data

        length = len(encrypted_data)
        counter = Counter(encrypted_data)
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)

        img_original = cv2.imread(image_path, 0)  # Charger l'image en niveaux de gris

        # Calcul de l'entropie pour l'image originale
        hist_original = np.histogram(img_original.flatten(), bins=256, range=[0, 256], density=True)[0]
        entropy_original = -np.sum(hist_original * np.log2(hist_original + 1e-10))

        entropieO_label.config(text=f'Entropie de limage originale : {entropy_original}')
        entropie_label.config(text=f'Entropie des données chiffrées : {entropy} ')


    menu_frame = tk.Frame(main_frame)
    main_frame.configure(bg='#87CEEB')

    # Création des styles
    style = ttk.Style()
    style.configure("Custom.TButton", foreground="#555555", background="blue")

    # Création des widgets avec des styles personnalisés
    choose_button = ttk.Button(main_frame, text="Choisir une Image", command=choose_image, style="Custom.TButton")
    choose_button.place(x=100, y=50)

    compress_button = ttk.Button(main_frame, text="Compress Image", command=compress_image, style="Custom.TButton")
    compress_button.place(x=400, y=50)

    evaluate_button = ttk.Button(main_frame, text="Évaluer les Performances", command=evaluate_performance,style="Custom.TButton")
    evaluate_button.place(x=700, y=50)

    # Labels pour les images
    original_label = tk.Label(main_frame, text="Original Image")
    original_label.place(x=100, y=100)
    original_image_label = tk.Label(main_frame, highlightthickness=2, highlightbackground="black")
    original_image_label.place(x=20, y=150)

    compressed_label = tk.Label(main_frame, text="Compressed Image")
    compressed_label.place(x=400, y=100)
    compressed_image_label = tk.Label(main_frame, highlightthickness=2, highlightbackground="black")
    compressed_image_label.place(x=360, y=150)

    # Labels pour les mesures de performance
    mse_label = tk.Label(main_frame, text="MSE: ")
    mse_label.place(x=700, y=150)

    psnr_label = tk.Label(main_frame, text="PSNR: ")
    psnr_label.place(x=700, y=200)

    ssim_label = tk.Label(main_frame, text="SSIM: ")
    ssim_label.place(x=700, y=250)

    mse_label.configure(fg='#000000', bg='#87CEEB')
    psnr_label.configure(fg='#000000', bg='#87CEEB')
    ssim_label.configure(fg='#000000', bg='#87CEEB')
    original_label.configure(fg='#000000', bg='#87CEEB')
    compressed_label.configure(fg='#000000', bg='#87CEEB')

    crypt_button = ttk.Button(main_frame, text="Crypter l'image compressée", command=crypte_image, style="Custom.TButton")
    crypt_button.place(x=150, y=380)

    Histogramme = ttk.Button(main_frame, text="Histogramme", command=histogram,style="Custom.TButton")
    Histogramme.place(x=600, y=400)
    Corellation = ttk.Button(main_frame, text="Corellation", command=dig_correlation,style="Custom.TButton")
    Corellation.place(x=600, y=450)
    Entropie = ttk.Button(main_frame, text="Entropie", command=calculate_entropy,style="Custom.TButton")
    Entropie.place(x=600, y=500)

    entropieO_label = tk.Label(main_frame, text="Entropie de l'image originale: ")
    entropieO_label.place(x=600, y=550)

    entropie_label = tk.Label(main_frame, text="Entropie des données chiffrées: ")
    entropie_label.place(x=600, y=600)

    entropieO_label.configure(fg='#000000', bg='#87CEEB')
    entropie_label.configure(fg='#000000', bg='#87CEEB')


    Crpted_label = tk.Label(main_frame, text="Crpted Image")
    Crpted_label.place(x=190, y=430)
    Crpted_image_label = tk.Label(main_frame, highlightthickness=2, highlightbackground="black")
    Crpted_image_label.place(x=100, y=460)

    Crpted_label.configure(fg='#000000', bg='#87CEEB')

    menu_frame.pack(pady=20)

MSET = 0
psnrT = 0
ssim_valueT = 0
i = 0
def contact_page():



    def evaluate_performance(image_path, compressed_image_path):
        global MSET, psnrT, ssim_valueT, i

        i += 1

        if image_path is None or compressed_image_path is None:
            print(
                "Erreur : Impossible d'évaluer les performances. Assurez-vous que les images sont chargées et compressées.")
            return

        original_image = cv2.imread(image_path)
        compressed_image = cv2.imread(compressed_image_path)

        # Convertir les images en niveaux de gris
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2GRAY)

        # Calculer les métriques de performance
        mse = np.mean((original_image - compressed_image) ** 2)
        MSET += mse

        # Calculer le PSNR (Peak Signal-to-Noise Ratio)
        psnr = 20 * math.log10(255 / math.sqrt(mse))
        psnrT += psnr

        # Calculer l'indice SSIM (Structural Similarity Index)
        ssim_value, _ = ssim(original_gray, compressed_gray, full=True)
        ssim_valueT += ssim_value


        # Assurez-vous que la sortie console n'est pas bloquée
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Affiche_performance():
        global MSET, psnrT, ssim_valueT, i

        if i == 0:
            print("Aucune évaluation de performance effectuée.")
            return

        mse_avg = MSET / i if i > 0 else 0
        psnr_avg = psnrT / i if i > 0 else 0
        ssim_avg = ssim_valueT / i if i > 0 else 0

        mse_label.config(text=f'MSE: {mse_avg}')
        psnr_label.config(text=f'PSNR: {psnr_avg} dB')
        ssim_label.config(text=f'SSIM: {ssim_avg}')

    def select_video():
        global path
        path = filedialog.askopenfilename()
        if path:
            cap = cv2.VideoCapture(path)
            show_video(cap)

    def show_video(cap):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (200, 200))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            original_image_label.config(image=img)
            original_image_label.image = img
            original_image_label.after(10, show_video, cap)
        else:
            cap.release()

    def catvidio():
        global path
        # Nom du dossier où vous voulez sauvegarder les images
        output_folder = 'output_images'

        # Création du dossier s'il n'existe pas
        os.makedirs(output_folder, exist_ok=True)

        # Ouvrir la vidéo
        video_capture = cv2.VideoCapture(path)

        # Compteur pour nommer les images
        image_count = 0

        while True:
            # Lecture d'une image de la vidéo
            ret, frame = video_capture.read()

            if not ret:
                break  # Fin de la vidéo

            # Nom de l'image à enregistrer avec chemin complet vers le dossier de sortie
            image_name = os.path.join(output_folder,
                                      f"frame_{image_count}.png")  # Vous pouvez choisir le format d'image souhaité (PNG, JPG, etc.)

            # Enregistrement de l'image
            cv2.imwrite(image_name, frame)

            # Incrément du compteur pour le nommage de l'image suivante
            image_count += 1

        # Libération des ressources
        video_capture.release()

    def compress_image(image_path, output_folder):
        if image_path:
            # Charger l'image en couleur
            original_image = cv2.imread(image_path)

            # Appliquer la transformation DWT à chaque canal de couleur
            compressed_image = np.zeros_like(original_image, dtype=np.float32)
            for i in range(3):  # Pour les canaux Rouge, Vert et Bleu
                channel = original_image[:, :, i]

                # Appliquer la transformation DWT
                coeffs = pywt.dwt2(channel, 'bior1.3')
                cA, (cH, cV, cD) = coeffs

                # Quantification - Ajuster la valeur pour le niveau de compression souhaité
                threshold = 1
                cA, cH, cV, cD = (pywt.threshold(c, threshold, mode='soft') for c in (cA, cH, cV, cD))

                # Reconstruire le canal
                compressed_channel = pywt.idwt2((cA, (cH, cV, cD)), 'bior1.3')
                compressed_image[:, :, i] = compressed_channel.astype(np.float32)

            # Convertir les valeurs à l'intervalle [0, 255]
            compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

            # Obtenez le nom du fichier
            filename = os.path.basename(image_path)
            # Créez le chemin de l'image compressée pour l'enregistrer
            compressed_image_path = os.path.join(output_folder, f"compressed_{filename}")

            # Enregistrer l'image compressée avec compression JPEG
            quality = 70  # Ajuster la qualité selon vos préférences (0-100)
            success = cv2.imwrite(compressed_image_path, compressed_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

            if success:
                evaluate_performance(image_path, compressed_image_path)
                return compressed_image_path
            else:
                return None
        else:
            return None

    def compress_images_in_folder():
        # Vérifiez si le dossier de sortie existe, sinon, créez-le
        if not os.path.exists("output_folder"):
            os.makedirs("output_folder")

        # Parcours de tous les fichiers du dossier d'entrée
        for file in os.listdir("output_images"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join("output_images", file)
                compressed_image_path = compress_image(image_path, "output_folder")
                if compressed_image_path:
                    print(f"Image compressée enregistrée à : {compressed_image_path}")
                else:
                    print(f"Erreur lors de la compression de l'image : {image_path}")

    def addvideo():
        input_folder = 'output_folder'

        # Nom du fichier vidéo de sortie
        output_video = 'compression_video.mp4'

        # Liste des noms de fichiers d'images dans le dossier
        image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                       os.path.isfile(os.path.join(input_folder, f))]

        # Trier les noms de fichiers dans l'ordre
        image_files.sort()

        # Déterminer les dimensions de la première image pour créer la vidéo avec les mêmes dimensions
        first_image = cv2.imread(image_files[0])
        height, width, layers = first_image.shape

        # Initialiser le codec vidéo et le VideoWriter
        fourcc = cv2.VideoWriter_fourcc(
            *'mp4v')  # Vous pouvez choisir le codec approprié pour le type de vidéo souhaité
        video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

        # Ajouter chaque image à la vidéo
        for image_file in image_files:
            img = cv2.imread(image_file)
            video.write(img)

        # Libérer la ressource de la vidéo et fermer le fichier
        video.release()

        cap1 = cv2.VideoCapture(output_video)
        show_video(cap1)
        ret, frame = cap1.read()
        if ret:
            frame = cv2.resize(frame, (200, 200))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            compressed_image_label.config(image=img)
            compressed_image_label.image = img
            compressed_image_label.after(10, show_video, cap1)
        else:
            cap1.release()

    def Compression_Video():
        catvidio()
        compress_images_in_folder()
        addvideo()

    def addvideochiffre():
        input_folder = 'chiffrement_sauvegarde'

        # Nom du fichier vidéo de sortie
        output_video = 'Chiffrement_video.mp4'

        # Liste des noms de fichiers d'images dans le dossier
        image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                       os.path.isfile(os.path.join(input_folder, f))]

        # Trier les noms de fichiers dans l'ordre
        image_files.sort()

        # Déterminer les dimensions de la première image pour créer la vidéo avec les mêmes dimensions
        first_image = cv2.imread(image_files[0])
        height, width, layers = first_image.shape

        # Initialiser le codec vidéo et le VideoWriter
        fourcc = cv2.VideoWriter_fourcc(
            *'mp4v')  # Vous pouvez choisir le codec approprié pour le type de vidéo souhaité
        video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

        # Ajouter chaque image à la vidéo
        for image_file in image_files:
            img = cv2.imread(image_file)
            video.write(img)

        # Libérer la ressource de la vidéo et fermer le fichier
        video.release()

        cap1 = cv2.VideoCapture(output_video)
        show_video(cap1)
        ret, frame = cap1.read()
        if ret:
            frame = cv2.resize(frame, (200, 200))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            Crpted_image_label.config(image=img)
            Crpted_image_label.image = img
            Crpted_image_label.after(10, show_video, cap1)
        else:
            cap1.release()


    def chiffrement_vidio():
        global encrypted_data
        # Génération de la clé ECC
        private_key = ec.generate_private_key(ec.SECP256R1())

        # Récupération de la clé publique
        public_key = private_key.public_key()

        # Sérialisation de la clé publique
        serialized_public = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Répertoire contenant les images à chiffrer
        input_directory = 'output_folder'  # Remplace par ton chemin

        counter = 1  # Initialisation du compteur

        for filename in os.listdir(input_directory):
            if filename.endswith('.png') or filename.endswith('.jpg'):  # Filtrer les images
                image_path = os.path.join(input_directory, filename)

                # Récupération de l'image
                image = Image.open(image_path)

                # Conversion de l'image en bytes
                img_byte_array = io.BytesIO()
                image.save(img_byte_array, format=image.format)
                img_bytes = img_byte_array.getvalue()

                # Chiffrement de l'image avec AES
                key = ec.generate_private_key(ec.SECP256R1())
                shared_key = private_key.exchange(ec.ECDH(), key.public_key())
                aes_key = shared_key[:32]  # Utilisation des 32 premiers octets comme clé AES

                # Padding des données de l'image
                padder = padding.PKCS7(algorithms.AES.block_size).padder()
                padded_data = padder.update(img_bytes) + padder.finalize()

                # Initialisation du chiffrement AES en mode CBC
                iv = os.urandom(16)  # Remplace par un vecteur d'initialisation approprié
                cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
                encryptor = cipher.encryptor()

                # Chiffrement des données
                encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

                # Sauvegarde de l'image chiffrée

                # Convertir les données chiffrées en une représentation d'image
                encrypted_image = Image.frombytes('RGB', (100, 100),
                                                  encrypted_data)  # Remplace (100, 100) par la taille de l'image

                encrypted_image.save(f'chiffrement_sauvegarde/image_chiffree{counter}.png')
                counter += 1

        # Sauvegarde de la clé publique
        with open('cle_publique.pem', 'wb') as file:
            file.write(serialized_public)

        addvideochiffre()

    def average_histogram_for_folders():
        original_histogram = Counter()
        encrypted_histogram = Counter()
        total_original_images = 0
        total_encrypted_images = 0

        # Calculer l'histogramme moyen pour les images originales
        for filename in os.listdir('output_images'):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join('output_images', filename)
                img_original = cv2.imread(image_path, 0)  # Charger l'image en niveaux de gris
                hist_original = Counter(img_original.flatten())

                # Cumuler les valeurs des histogrammes originaux
                original_histogram += hist_original
                total_original_images += 1

        # Calculer l'histogramme moyen pour les images chiffrées
        for filename in os.listdir('output_images'):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join('output_images', filename)
                img_encrypted = cv2.imread(image_path, 0)  # Charger l'image en niveaux de gris
                hist_encrypted = Counter(img_encrypted.flatten())

                # Cumuler les valeurs des histogrammes chiffrés
                encrypted_histogram += hist_encrypted
                total_encrypted_images += 1

        # Calculer la moyenne des histogrammes pour les images originales et chiffrées
        average_original_histogram = {key: value / total_original_images for key, value in original_histogram.items()}
        average_encrypted_histogram = {key: value / total_encrypted_images for key, value in
                                       encrypted_histogram.items()}

        # Création du graphique pour l'histogramme moyen des images originales
        plt.bar(average_original_histogram.keys(), average_original_histogram.values(), alpha=0.5, label='Original')
        # Ajout des données pour l'histogramme moyen des images chiffrées
        plt.bar(average_encrypted_histogram.keys(), average_encrypted_histogram.values(), alpha=0.5, label='Chiffré')
        plt.xlabel('Valeur des pixels')
        plt.ylabel('Fréquence moyenne')
        plt.title('Comparaison des histogrammes moyens')
        plt.legend()
        plt.show()

    def histogram():

        global encrypted_data

        img_original = cv2.imread('output_images/frame_0.png', 0)  # Charger l'image en niveaux de gris
        hist_original = Counter(img_original.flatten())

        # Histogramme des données chiffrées
        histogram_encrypted = Counter(encrypted_data)

        # Création du graphique
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(hist_original.keys(), hist_original.values())
        plt.xlabel('Valeur des pixels')
        plt.ylabel('Fréquence')
        plt.title('Histogramme de vedio originale')

        plt.subplot(1, 2, 2)
        plt.bar(histogram_encrypted.keys(), histogram_encrypted.values())
        plt.xlabel('Valeur des octets')
        plt.ylabel('Fréquence')
        plt.title('Histogramme des octets chiffrés')

        plt.tight_layout()
        plt.show()



    def calculate_correlation(data):
        correlation = sum(data[i] == data[i + 1] for i in range(len(data) - 1)) / (len(data) - 1)
        return correlation

    def dig_correlation():

        global encrypted_data

        # Chargement de l'image originale
        original_image = Image.open('chiffrement_sauvegarde/image_chiffree1.png')
        original_gray = original_image.convert('L')  # Conversion en niveaux de gris
        original_array = np.array(original_gray)

        # Calcul de la corrélation pour l'image originale
        correlation_original = np.corrcoef(original_array[0:-1, 0:-1].flatten(), original_array[1:, 1:].flatten())[0, 1]

        # Conversion de l'image chiffrée en niveaux de gris

        correlation = calculate_correlation(encrypted_data)
        encoded_data = base64.b64encode(encrypted_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        encoded_data = base64.b64encode(encoded_data)
        image_size = (100, 100)  # Mettez la taille de l'image ici
        encrypted_image = Image.frombytes('RGB', image_size, encoded_data)

        # Convertir l'image en niveaux de gris pour simplifier la corrélation
        encrypted_gray = encrypted_image.convert('L')
        encrypted_array = np.array(encrypted_gray)

        # Affichage des diagrammes de dispersion
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(original_array[0:-1, 0:-1].flatten(), original_array[1:, 1:].flatten(), s=1, alpha=0.5)
        plt.title(f'Diagramme de corrélation - vedio originale\nCorrélation : {correlation_original:.4f}')
        plt.xlabel('Pixel (i, j)')
        plt.ylabel('Pixel (i+1, j+1)')

        plt.subplot(1, 2, 2)
        plt.scatter(encrypted_array[0:-1, 0:-1].flatten(), encrypted_array[1:, 1:].flatten(), s=1, alpha=0.5)
        plt.title(f'Diagramme de corrélation\nCorrélation : {correlation:.4f}')
        plt.xlabel('Pixel (i, j)')
        plt.ylabel('Pixel (i+1, j+1)')

        plt.tight_layout()
        plt.show()

    def calculate_entropy():
        global encrypted_data

        length = len(encrypted_data)
        counter = Counter(encrypted_data)
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)

        img_original = cv2.imread('chiffrement_sauvegarde/image_chiffree1.png', 0)  # Charger l'image en niveaux de gris

        # Calcul de l'entropie pour l'image originale
        hist_original = np.histogram(img_original.flatten(), bins=256, range=[0, 256], density=True)[0]
        entropy_original = -np.sum(hist_original * np.log2(hist_original + 1e-10))

        entropieO_label.config(text=f'Entropie de limage originale : {entropy_original}')
        entropie_label.config(text=f'Entropie des données chiffrées : {entropy} ')


    contact_frame = tk.Frame(main_frame)


    main_frame.configure(bg='#87CEEB')

    # Création des styles
    style = ttk.Style()
    style.configure("Custom.TButton", foreground="#555555", background="blue")

    # Création des widgets avec des styles personnalisés
    choose_button = ttk.Button(main_frame, text="Sélectionner une vidéo",command=select_video, style="Custom.TButton")
    choose_button.place(x=100, y=50)

    compress_button = ttk.Button(main_frame, text="Compress vidéo" , command=Compression_Video, style="Custom.TButton")
    compress_button.place(x=400, y=50)

    evaluate_button = ttk.Button(main_frame, text="Évaluer les Performances", command=Affiche_performance,style="Custom.TButton")
    evaluate_button.place(x=700, y=50)

    # Labels pour les images
    original_label = tk.Label(main_frame, text="Original vidéo")
    original_label.place(x=100, y=100)
    original_image_label = tk.Label(main_frame, highlightthickness=2, highlightbackground="black")
    original_image_label.place(x=20, y=150)

    compressed_label = tk.Label(main_frame, text="Compressed vidéo")
    compressed_label.place(x=400, y=100)
    compressed_image_label = tk.Label(main_frame, highlightthickness=2, highlightbackground="black")
    compressed_image_label.place(x=360, y=150)

    # Labels pour les mesures de performance
    mse_label = tk.Label(main_frame, text="MSE: ")
    mse_label.place(x=700, y=150)

    psnr_label = tk.Label(main_frame, text="PSNR: ")
    psnr_label.place(x=700, y=200)

    ssim_label = tk.Label(main_frame, text="SSIM: ")
    ssim_label.place(x=700, y=250)

    mse_label.configure(fg='#000000', bg='#87CEEB')
    psnr_label.configure(fg='#000000', bg='#87CEEB')
    ssim_label.configure(fg='#000000', bg='#87CEEB')
    original_label.configure(fg='#000000', bg='#87CEEB')
    compressed_label.configure(fg='#000000', bg='#87CEEB')

    crypt_button = ttk.Button(main_frame, text="Crypter vidéo compressée", command=chiffrement_vidio, style="Custom.TButton")
    crypt_button.place(x=150, y=380)

    Histogramme = ttk.Button(main_frame, text="Histogramme", command=histogram,style="Custom.TButton")
    Histogramme.place(x=600, y=400)
    Corellation = ttk.Button(main_frame, text="Corellation", command=dig_correlation, style="Custom.TButton")
    Corellation.place(x=600, y=450)
    Entropie = ttk.Button(main_frame, text="Entropie", command=calculate_entropy, style="Custom.TButton")
    Entropie.place(x=600, y=500)

    entropieO_label = tk.Label(main_frame, text="Entropie de vidéo originale: ")
    entropieO_label.place(x=600, y=550)

    entropie_label = tk.Label(main_frame, text="Entropie des vidéo chiffrées: ")
    entropie_label.place(x=600, y=600)

    entropieO_label.configure(fg='#000000', bg='#87CEEB')
    entropie_label.configure(fg='#000000', bg='#87CEEB')


    Crpted_label = tk.Label(main_frame, text="Crpted vidéo")
    Crpted_label.place(x=190, y=430)
    Crpted_image_label = tk.Label(main_frame, highlightthickness=2, highlightbackground="black")
    Crpted_image_label.place(x=100, y=460)

    Crpted_label.configure(fg='#000000', bg='#87CEEB')


    contact_frame.pack(pady=20)




def hide_indicateur():
    home_indicate.config(bg='#0A043C')
    menu_indicate.config(bg='#0A043C')
    contact_indicate.config(bg='#0A043C')

def delete_pages():
    for frame in main_frame.winfo_children():
        frame.destroy()



def indicate(lb , page):
    hide_indicateur()
    lb.config(bg='#158aff')
    delete_pages()
    page()


options_frame = tk.Frame(root, bg='#0A043C')

# Charger l'image de logo
logo_image = Image.open("FST.jpg")  # Remplacez par le chemin de votre image
logo_image = logo_image.resize((150, 50))  # Redimensionnez l'image selon vos besoins

# Convertir l'image pour l'affichage dans tkinter
logo_tk = ImageTk.PhotoImage(logo_image)

# Créer un Label pour afficher l'image du logo
logo_label = tk.Label(options_frame, image=logo_tk, bg='#0A043C')
logo_label.place(x=3, y=10)  # Ajustez les coordonnées selon votre mise en page

# Assurez-vous de garder une référence à l'image pour éviter qu'elle ne soit effacée par le garbage collector
logo_label.image = logo_tk


separator_line = tk.Frame(options_frame, bg='white')
separator_line.place(x=0, y=120, width=150, height=2)

separator_line = tk.Frame(options_frame, bg='white')
separator_line.place(x=0, y=210, width=150, height=2)

separator_line = tk.Frame(options_frame, bg='white')
separator_line.place(x=0, y=300, width=150, height=2)

separator_line = tk.Frame(options_frame, bg='white')
separator_line.place(x=0, y=400, width=150, height=2)

home_bnt = tk.Button(options_frame, text='Text' , font=('Bold',15), fg='#158aff', bd=0, bg='#0A043C',command=lambda:indicate(home_indicate,home_page))
home_bnt.place(x=30, y=150)
home_indicate = tk.Label(options_frame ,text='',bg='#0A043C')
home_indicate.place(x=3, y=150, width=5, height=36)

menu_bnt = tk.Button(options_frame, text='Image' , font=('Bold',15), fg='#158aff', bd=0, bg='#0A043C',command=lambda:indicate(menu_indicate,menu_page))
menu_bnt.place(x=30, y=240)
menu_indicate = tk.Label(options_frame ,text='',bg='#0A043C')
menu_indicate.place(x=3, y=240, width=5, height=36)

contact_bnt = tk.Button(options_frame, text='Vidéo' , font=('Bold',15), fg='#158aff', bd=0, bg='#0A043C',command=lambda:indicate(contact_indicate,contact_page))
contact_bnt.place(x=30, y=330)
contact_indicate = tk.Label(options_frame ,text='',bg='#0A043C')
contact_indicate.place(x=3, y=330, width=5, height=36)

options_frame.pack(side=tk.LEFT)
options_frame.pack_propagate(False)
options_frame.configure(width=150, height=700)







main_frame = tk.Frame(root , highlightbackground='black',highlightthickness=2)

main_frame.pack(side=tk.LEFT)
main_frame.pack_propagate(False)
main_frame.configure(height=700, width=1000)

root.mainloop()