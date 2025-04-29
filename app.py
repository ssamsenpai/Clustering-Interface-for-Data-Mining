import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    print("Veuillez installer sklearn_extra : pip install scikit-learn-extra")

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparaison de méthodes de clustering")
        self.root.geometry("1400x900")

        self.data = None
        self.cleaned_data = None
        self.results = []

        self.data_choice = tk.StringVar(value="Fichier CSV")
        self.algo_choice = tk.StringVar(value="KMeans")
        self.n_clusters = tk.IntVar(value=3)
        self.eps = tk.DoubleVar(value=0.5)
        self.min_samples = tk.IntVar(value=5)

        self.setup_layout()

    def setup_layout(self):
        # ===== Scrollable Canvas Setup =====
        self.canvas = tk.Canvas(self.root)
        self.scroll_y = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)
        self.scroll_y.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.frame_main = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame_main, anchor="nw")
        self.frame_main.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # ===== Sections =====
        self.create_data_section()
        self.create_parameters_section()
        self.create_buttons_section()
        self.create_results_section()
        self.create_plot_section()

    def create_data_section(self):
        frame = ttk.LabelFrame(self.frame_main, text="Chargement des Données", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame, text="Source:").pack(side="left", padx=5)
        data_menu = ttk.Combobox(frame, textvariable=self.data_choice, values=["Fichier CSV", "Iris Dataset"], width=20)
        data_menu.pack(side="left", padx=5)

        tk.Button(frame, text="Charger", command=self.load_data).pack(side="left", padx=10)

    def create_parameters_section(self):
        frame = ttk.LabelFrame(self.frame_main, text="Paramètres de Clustering", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Algorithm choice
        ttk.Label(frame, text="Algorithme:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        algo_menu = ttk.Combobox(frame, textvariable=self.algo_choice, values=["KMeans", "KMedoids", "Agglomerative", "DBSCAN"], width=20)
        algo_menu.grid(row=0, column=1, padx=5, pady=5)

        # Number of clusters
        ttk.Label(frame, text="Nombre de Clusters (k):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(frame, textvariable=self.n_clusters, width=10).grid(row=1, column=1, padx=5, pady=5)

        # DBSCAN specific parameters
        ttk.Label(frame, text="EPS (DBSCAN):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(frame, textvariable=self.eps, width=10).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Min Samples (DBSCAN):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(frame, textvariable=self.min_samples, width=10).grid(row=3, column=1, padx=5, pady=5)

    def create_buttons_section(self):
        frame = ttk.Frame(self.frame_main, padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Button(frame, text="Lancer Clustering", command=self.run_clustering).pack(side="left", padx=10)
        tk.Button(frame, text="Afficher Courbe Elbow", command=self.plot_elbow).pack(side="left", padx=10)

    def create_results_section(self):
        self.frame_table = ttk.LabelFrame(self.frame_main, text="Tableau de Résultats", padding=10)
        self.frame_table.pack(fill="both", expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(self.frame_table, columns=["Algorithm","Clusters","Silhouette Score", "Davies-Bouldin Score"], show='headings')
        self.tree.heading("Algorithm", text="Algorithm")
        self.tree.heading("Clusters", text="Clusters")
        self.tree.heading("Silhouette Score", text="Silhouette Score")
        self.tree.heading("Davies-Bouldin Score", text="Davies-Bouldin Score")
        self.tree.column("Algorithm", width=200)
        self.tree.column("Clusters", width=100)
        self.tree.column("Silhouette Score", width=150)
        self.tree.column("Davies-Bouldin Score", width=200)
        self.tree.pack(fill="both", expand=True)

    def create_plot_section(self):
        self.frame_results = ttk.LabelFrame(self.frame_main, text="Visualisations et Scores", padding=10)
        self.frame_results.pack(fill="both", expand=True, padx=10, pady=5)

    def load_data(self):
        choice = self.data_choice.get()
        if choice == "Fichier CSV":
            file_path = filedialog.askopenfilename()
            if file_path:
                self.data = pd.read_csv(file_path)
                print("Données CSV chargées.")
        elif choice == "Iris Dataset":
            iris = load_iris()
            self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
            print("Dataset Iris chargé.")
        
        self.clean_data()

    def clean_data(self):
        if self.data is not None:
            df = self.data.select_dtypes(include=[np.number]).fillna(method='ffill').fillna(method='bfill')
            self.cleaned_data = df
            print(f"Données nettoyées: {df.shape}")

    def run_clustering(self):
        if self.cleaned_data is None:
            print("Veuillez charger les données d'abord.")
            return

        self.clear_results()

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.cleaned_data)

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        algo = self.algo_choice.get()
        num_clusters = self.n_clusters.get()

        if algo == "KMeans":
            model = KMeans(n_clusters=num_clusters, random_state=0)
        elif algo == "KMedoids":
            model = KMedoids(n_clusters=num_clusters, random_state=0)
        elif algo == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=num_clusters)
        elif algo == "DBSCAN":
            model = DBSCAN(eps=self.eps.get(), min_samples=self.min_samples.get())
        else:
            print("Algorithme non reconnu.")
            return

        clusters = model.fit_predict(data_scaled)

        try:
            silhouette = silhouette_score(data_scaled, clusters)
            db_score = davies_bouldin_score(data_scaled, clusters)
            self.results.append([algo, num_clusters, f"{silhouette:.3f}", f"{db_score:.3f}"])
        except Exception as e:
            silhouette = db_score = None
            print(f"Erreur lors du calcul des scores: {e}")

        self.update_table()
        self.display_metrics(silhouette, db_score)
        self.display_plot(data_pca, clusters, algo)

        if algo == "Agglomerative":
            self.display_dendrogram(data_scaled)

    def clear_results(self):
        for widget in self.frame_results.winfo_children():
            widget.destroy()

    def update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for result in self.results:
            self.tree.insert("", "end", values=result)

    def display_metrics(self, silhouette, db_score):
        frame = tk.Frame(self.frame_results)
        frame.pack(fill="x", pady=10)

        if silhouette is not None and db_score is not None:
            text = f"Silhouette Score: {silhouette:.3f} | Davies-Bouldin Score: {db_score:.3f}"
        else:
            text = "Scores non disponibles."

        tk.Label(frame, text=text, font=("Arial", 12, "bold")).pack()

    def display_plot(self, data_pca, clusters, algo):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='rainbow')
        ax.set_title(f"Clustering avec {algo}")
        plt.colorbar(scatter, ax=ax)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_results)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def display_dendrogram(self, data_scaled):
        linked = linkage(data_scaled, method='ward')
        fig, ax = plt.subplots(figsize=(8, 6))
        dendrogram(linked, ax=ax)
        ax.set_title("Dendrogramme Agglomerative")

        canvas = FigureCanvasTkAgg(fig, master=self.frame_results)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def plot_elbow(self):
        if self.cleaned_data is None:
            print("Veuillez charger les données avant.")
            return

        self.clear_results()

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.cleaned_data)

        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=0)
            model.fit(data_scaled)
            inertias.append(model.inertia_)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(k_range, inertias, marker='o')
        ax.set_title("Méthode du Coude")
        ax.set_xlabel("Nombre de Clusters")
        ax.set_ylabel("Inertie")
        plt.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_results)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()
