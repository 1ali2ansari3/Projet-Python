import tkinter as tk
from tkinter import messagebox

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

# Create the main window
window = tk.Tk()
window.title("Encryption/Decryption Interface")

# Create and place labels, entry widgets, and buttons
label_n = tk.Label(window, text="Enter the prime number 'P':")
label_n.pack()

entry_n = tk.Entry(window)
entry_n.pack()

label_d = tk.Label(window, text="Enter the random number 'x' (Private key of Sender):")
label_d.pack()

entry_d = tk.Entry(window)
entry_d.pack()

label_k = tk.Label(window, text="Enter the random number 'g' (Encryption key):")
label_k.pack()

entry_k = tk.Entry(window)
entry_k.pack()

label_message = tk.Label(window, text="Enter the message to be sent:")
label_message.pack()

entry_message = tk.Entry(window)
entry_message.pack()

encrypt_button = tk.Button(window, text="Encrypt/Decrypt", command=encrypt_decrypt)
encrypt_button.pack()

result_display = tk.Text(window, height=10, width=50)
result_display.pack()

# Run the Tkinter event loop
window.mainloop()
