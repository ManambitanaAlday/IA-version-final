import os
import json
import urllib.request
import urllib.error
import threading
import math
import re
import unicodedata
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox

CLEANED_JSON_DIR = "./json_cleaned"
OLLAMA_BASE_URL = "http://localhost:11434"


def normalize_text(text):
    if not text:
        return ""
    text_nfkd = unicodedata.normalize('NFD', str(text))
    return ''.join(c for c in text_nfkd if unicodedata.category(c) != 'Mn').lower()


def format_substituts_to_string(subs):
    if not subs:
        return "Aucun substitut répertorié."
    
    if isinstance(subs, list):
        cleaned_subs = []
        for item in subs:
            if isinstance(item, dict):
                nom_sub = item.get('nom', item.get('nom_substitut', ''))
                raison = item.get('raison', '')
                if nom_sub:
                    entry = f"{nom_sub} ({raison})" if raison else nom_sub
                    cleaned_subs.append(entry)
            else:
                cleaned_subs.append(str(item))
        return ", ".join(cleaned_subs) if cleaned_subs else "Aucun substitut répertorié."
    elif isinstance(subs, dict):
        return str(subs.get('nom', list(subs.values())[0] if subs else ''))
    
    return str(subs)


class AutoOllamaManager:
    @staticmethod
    def get_installed_models():
        """Détecte automatiquement la liste des modèles installés sur Ollama."""
        url = f"{OLLAMA_BASE_URL}/api/tags"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []


class NativeFastRAG:
    def __init__(self):
        self.docs = []
        self.vocab = {}
        self.tfidf_matrix = []

    def tokenize(self, text):
        return re.findall(r'\w+', normalize_text(text))

    def index_documents(self, meds_database):
        self.docs = []
        raw_texts = []
        
        for med in meds_database:
            if isinstance(med, dict):
                subs_str = format_substituts_to_string(med.get('substituts', []))
                maladies = ", ".join(med.get('maladies_associees', [])) if isinstance(med.get('maladies_associees'), list) else str(med.get('maladies_associees', ''))
                ci = ", ".join(med.get('contre_indications', [])) if isinstance(med.get('contre_indications'), list) else str(med.get('contre_indications', ''))
                effets = ", ".join(med.get('effets_secondaires', [])) if isinstance(med.get('effets_secondaires'), list) else str(med.get('effets_secondaires', ''))

                text = (
                    f"Nom: {med.get('nom', '')} | Nom Générique: {med.get('nom_generique', '')} | "
                    f"Famille: {med.get('famille', '')} | Dosage: {med.get('dosage', '')} | "
                    f"Indications / Maladies: {maladies} | "
                    f"Posologie: {med.get('posologie_standard', '')} | "
                    f"Contre-indications: {ci} | Effets secondaires: {effets} | "
                    f"Substituts: {subs_str}"
                )
                self.docs.append((text, med))
                raw_texts.append(text)

        tokenized_docs = [self.tokenize(t) for t in raw_texts]
        all_words = set(w for doc in tokenized_docs for w in doc)
        self.vocab = {word: i for i, word in enumerate(all_words)}
        
        num_docs = len(tokenized_docs)
        if num_docs == 0:
            return

        doc_freqs = Counter(w for doc in tokenized_docs for w in set(doc))
        idfs = {w: math.log((num_docs + 1) / (freq + 1)) + 1 for w, freq in doc_freqs.items()}

        self.tfidf_matrix = []
        for doc in tokenized_docs:
            tf = Counter(doc)
            doc_len = len(doc) or 1
            vec = {}
            for word, count in tf.items():
                if word in self.vocab:
                    vec[self.vocab[word]] = (count / doc_len) * idfs[word]
            self.tfidf_matrix.append(vec)

    def search(self, query, top_k=3):
        if not self.docs or not self.tfidf_matrix:
            return ""

        query_tokens = self.tokenize(query)
        query_vec = {}
        for w in query_tokens:
            if w in self.vocab:
                query_vec[self.vocab[w]] = query_tokens.count(w)

        if not query_vec:
            return ""

        def cosine_sim(vec_a, vec_b):
            dot = sum(val * vec_b.get(idx, 0.0) for idx, val in vec_a.items())
            norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
            norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        scores = [
            (cosine_sim(query_vec, doc_vec), self.docs[i][0])
            for i, doc_vec in enumerate(self.tfidf_matrix)
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [doc_text for score, doc_text in scores[:top_k] if score > 0.01]
        return "\n\n".join(results)


class PharmaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PharmaAssist - Assistant Pharmacie & Gestion de Stock")
        self.root.geometry("950x700")
        self.root.minsize(800, 600)

        self.colors = {
            "bg_dark": "#0f172a",
            "bg_card": "#1e293b",
            "primary": "#0d9488",
            "primary_hover": "#0f766e",
            "accent": "#38bdf8",
            "text_light": "#f8fafc",
            "text_muted": "#94a3b8",
            "system_msg": "#f59e0b"
        }

        self.root.configure(bg=self.colors["bg_dark"])

        self.meds_database = []
        self.rag = NativeFastRAG()
        self.installed_models = []
        self.is_ready = False
        self.pending_order = None

        self.apply_styles()
        self.setup_gui()
        self.append_chat("Système : Détection automatique des modèles Ollama installés...\n", "system")
        threading.Thread(target=self.async_init, daemon=True).start()

    def apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("TNotebook", background=self.colors["bg_dark"], borderwidth=0)
        style.configure("TNotebook.Tab", background=self.colors["bg_card"], foreground=self.colors["text_muted"],
                        padding=[15, 8], font=('Helvetica', 10, 'bold'), borderwidth=0)
        style.map("TNotebook.Tab", 
                  background=[("selected", self.colors["primary"])],
                  foreground=[("selected", self.colors["text_light"])])

        style.configure("TFrame", background=self.colors["bg_dark"])

        style.configure("Treeview", 
                        background=self.colors["bg_card"],
                        foreground=self.colors["text_light"],
                        fieldbackground=self.colors["bg_card"],
                        rowheight=30,
                        font=('Helvetica', 10),
                        borderwidth=0)
        style.configure("Treeview.Heading", 
                        background=self.colors["primary"],
                        foreground=self.colors["text_light"],
                        font=('Helvetica', 10, 'bold'),
                        relief="flat")
        style.map("Treeview", background=[('selected', self.colors["primary_hover"])])

        style.configure("Primary.TButton", 
                        background=self.colors["primary"], 
                        foreground=self.colors["text_light"],
                        font=('Helvetica', 10, 'bold'),
                        borderwidth=0,
                        focusthickness=0,
                        padding=[15, 8])
        style.map("Primary.TButton", background=[("active", self.colors["primary_hover"])])

    def async_init(self):
        try:
            # 1. Détection automatique des modèles
            self.installed_models = AutoOllamaManager.get_installed_models()
            
            # Mise à jour du menu déroulant dans l'UI
            self.root.after(0, self.update_model_combobox)

            # 2. Chargement des JSON et RAG
            self.load_cleaned_data()
            self.rag.index_documents(self.meds_database)
            self.root.after(0, self.refresh_stock_view)
            
            self.is_ready = True
            
            if self.installed_models:
                self.append_chat(f"Système : [✓] {len(self.installed_models)} modèle(s) trouvé(s). Choisissez dans le menu en haut à droite.\n\n", "system")
            else:
                self.append_chat("Système : [!] Aucun modèle trouvé sur Ollama. Vérifiez qu'Ollama est démarré.\n\n", "system")
                
        except Exception as e:
            self.append_chat(f"Système : [X] Erreur d'initialisation : {e}\n\n", "system")

    def update_model_combobox(self):
        if self.installed_models:
            self.model_combobox['values'] = self.installed_models
            self.model_combobox.current(0)
        else:
            self.model_combobox['values'] = ["Aucun modèle trouvé"]
            self.model_combobox.current(0)

    def load_cleaned_data(self):
        self.meds_database = []
        if os.path.exists(CLEANED_JSON_DIR):
            for file in os.listdir(CLEANED_JSON_DIR):
                if file.endswith(".json"):
                    with open(os.path.join(CLEANED_JSON_DIR, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.meds_database.extend(data)
                        elif isinstance(data, dict):
                            self.meds_database.extend(data.get('medicaments', [data]))
        
        # Harmonisation de la clé de stock
        for med in self.meds_database:
            if isinstance(med, dict):
                if 'stock' not in med:
                    med['stock'] = med.get('stock_actuel', 20)

    def setup_gui(self):
        header_frame = tk.Frame(self.root, bg=self.colors["bg_card"], height=60)
        header_frame.pack(fill='x', side='top')
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(header_frame, text="PharmaAssist AI", font=('Helvetica', 16, 'bold'),
                             bg=self.colors["bg_card"], fg=self.colors["accent"])
        title_lbl.pack(side='left', padx=20)

        model_frame = tk.Frame(header_frame, bg=self.colors["bg_card"])
        model_frame.pack(side='right', padx=20)

        model_lbl = tk.Label(model_frame, text="Modèle LLM :", font=('Helvetica', 9, 'bold'),
                             bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        model_lbl.pack(side='left', padx=(0, 8))

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=20)
        self.model_combobox['values'] = ["Détection en cours..."]
        self.model_combobox.current(0)
        self.model_combobox.pack(side='right')

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=15, pady=15)

        # Chat Tab
        chat_frame = ttk.Frame(notebook)
        notebook.add(chat_frame, text=" Assistant Chatbot ")

        chat_container = tk.Frame(chat_frame, bg=self.colors["bg_card"], bd=1, relief="solid")
        chat_container.pack(fill='both', expand=True, padx=10, pady=(10, 5))

        self.chat_display = tk.Text(chat_container, state='disabled', wrap='word', 
                                    bg=self.colors["bg_card"], fg=self.colors["text_light"],
                                    font=('Segoe UI', 10), bd=0, highlightthickness=0, padx=15, pady=15)
        self.chat_display.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(chat_container, command=self.chat_display.yview)
        scrollbar.pack(side='right', fill='y')
        self.chat_display['yscrollcommand'] = scrollbar.set

        self.chat_display.tag_configure("system", foreground=self.colors["system_msg"], font=('Segoe UI', 9, 'italic'))
        self.chat_display.tag_configure("user", foreground=self.colors["accent"], font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_configure("bot", foreground=self.colors["text_light"], font=('Segoe UI', 10))

        input_frame = tk.Frame(chat_frame, bg=self.colors["bg_dark"])
        input_frame.pack(fill='x', padx=10, pady=10)

        self.user_entry = tk.Entry(input_frame, bg=self.colors["bg_card"], fg=self.colors["text_light"],
                                   insertbackground=self.colors["text_light"], font=('Segoe UI', 11),
                                   bd=1, relief="solid", highlightthickness=1, 
                                   highlightbackground=self.colors["primary"], highlightcolor=self.colors["accent"])
        self.user_entry.pack(side='left', fill='x', expand=True, ipady=8, padx=(0, 10))
        self.user_entry.bind("<Return>", lambda event: self.handle_chat())

        send_btn = ttk.Button(input_frame, text="Envoyer", style="Primary.TButton", command=self.handle_chat)
        send_btn.pack(side='right')

        # Stock Tab
        stock_frame = ttk.Frame(notebook)
        notebook.add(stock_frame, text=" Gestion de Stock ")

        stock_container = tk.Frame(stock_frame, bg=self.colors["bg_card"])
        stock_container.pack(fill='both', expand=True, padx=10, pady=10)

        self.stock_tree = ttk.Treeview(stock_container, columns=("Nom", "Famille", "Stock"), show='headings')
        self.stock_tree.heading("Nom", text="Nom du Médicament")
        self.stock_tree.heading("Famille", text="Famille / Catégorie")
        self.stock_tree.heading("Stock", text="Stock Disponible")
        
        self.stock_tree.column("Nom", width=350, minwidth=150)
        self.stock_tree.column("Famille", width=350, minwidth=150)
        self.stock_tree.column("Stock", width=150, minwidth=80, anchor="center")

        stock_scroll = ttk.Scrollbar(stock_container, orient="vertical", command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=stock_scroll.set)

        self.stock_tree.pack(side='left', fill='both', expand=True)
        stock_scroll.pack(side='right', fill='y')

    def refresh_stock_view(self):
        for item in self.stock_tree.get_children():
            self.stock_tree.delete(item)
        for med in self.meds_database:
            if isinstance(med, dict):
                self.stock_tree.insert("", "end", values=(
                    med.get('nom', 'N/A'),
                    med.get('famille', 'N/A'),
                    med.get('stock', 0)
                ))

    def parse_stock_intent(self, text):
        clean_text = normalize_text(text)
        match = re.search(r'(?:je veux|prends|achete|commander|besoin de|donne moi)?\s*(\d+)\s+([a-z0-9\s]+)', clean_text)
        if match:
            qty = int(match.group(1))
            search_name = match.group(2).strip()
            
            for med in self.meds_database:
                if isinstance(med, dict):
                    med_nom_clean = normalize_text(med.get('nom', ''))
                    if search_name in med_nom_clean or med_nom_clean in search_name:
                        return med, qty
        return None, 0

    def query_ollama(self, prompt):
        selected_model = self.model_var.get()
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = json.dumps({
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }).encode('utf-8')
        
        try:
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "Aucune réponse générée.")
        except Exception as e:
            return f"Erreur de communication avec le modèle '{selected_model}' ({e})."

    def handle_chat(self):
        user_text = self.user_entry.get().strip()
        if not user_text:
            return

        if not self.is_ready:
            messagebox.showwarning("Patientez", "L'initialisation est en cours...")
            return

        self.user_entry.delete(0, tk.END)
        self.append_chat(f"Vous : {user_text}\n", "user")

        # 1. Confirmation / Annulation d'achat
        if self.pending_order:
            clean_input = normalize_text(user_text)
            if any(k in clean_input for k in ["oui", "valider", "ok", "confirmer", "ouais"]):
                med = self.pending_order['med']
                qty = self.pending_order['qty']
                med['stock'] -= qty
                self.refresh_stock_view()
                self.append_chat(f"Assistant : [✓] Achat validé ! {qty} x {med['nom']} déduit(s). Stock restant : {med['stock']}.\n\n", "bot")
                self.pending_order = None
                return
            elif any(k in clean_input for k in ["non", "annuler", "stop"]):
                self.append_chat("Assistant : [X] Opération annulée. Le stock n'a pas été modifié.\n\n", "bot")
                self.pending_order = None
                return

        # 2. Traitement direct des commandes de Stock & Substituts
        med_found, requested_qty = self.parse_stock_intent(user_text)
        if med_found:
            current_stock = med_found.get('stock', 0)
            
            if current_stock >= requested_qty:
                self.pending_order = {'med': med_found, 'qty': requested_qty}
                self.append_chat(
                    f"Assistant : {med_found['nom']} est disponible en stock ({current_stock} unité(s) disponible(s)).\n"
                    f"👉 Voulez-vous valider l'achat de {requested_qty} x {med_found['nom']} ? (Répondez 'Oui' pour valider ou 'Non' pour annuler)\n\n",
                    "bot"
                )
            else:
                subs_text = format_substituts_to_string(med_found.get('substituts', []))
                self.append_chat(
                    f"Assistant : [!] Indisponible. Le stock de {med_found['nom']} est de {current_stock} (Demandé : {requested_qty}).\n"
                    f"💊 Substitut(s) disponible(s) dans la base : {subs_text}\n\n",
                    "bot"
                )
            return

        # 3. Traitement des questions médicales avec FastRAG + Prompt Strict
        def process():
            current_model = self.model_var.get()
            context = self.rag.search(user_text)
            
            prompt = (
                f"Vous êtes un assistant pharmacien professionnel et rigoureux.\n"
                f"RÈGLES D'OR :\n"
                f"1. Répondez STRICTEMENT en fonction des données fournies dans le CONTEXTE ci-dessous.\n"
                f"2. N'inventez AUCUN effet secondaire, indication ou contre-indication non présent dans le contexte.\n"
                f"3. Si l'information n'est pas dans le contexte, dites clairement : 'Je ne dispose pas de cette information dans la base de données.'\n\n"
                f"--- CONTEXTE BD ---\n{context if context else 'Aucune fiche trouvée.'}\n--------------------\n\n"
                f"Question du client : {user_text}\n"
                f"Réponse :"
            )
            response = self.query_ollama(prompt)
            self.append_chat(f"Assistant ({current_model}) : {response}\n\n", "bot")

        threading.Thread(target=process, daemon=True).start()

    def append_chat(self, msg, tag="bot"):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, msg, tag)
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = PharmaApp(root)
    root.mainloop()