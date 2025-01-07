import face_recognition
import numpy as np
import os
import cv2
from tqdm import tqdm

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from collections import defaultdict
import shutil



class ModelFaceRecognition:

    def __init__(self, path_raw_image):

        # Path du dossier initiale
        self.path_raw_image = path_raw_image
        # Path du dossier réduit
        self.path_reduced_image = "data/reduced"
        # Path du dossier des visages
        self.path_face_image = "data/faces"

        self.new_width = 800
        self.new_height = 600

        self.faces_data = []
    
    # Fonction pour réduire la taille des images de path_raw_image pour les transformer dans path_reduced_image
    def reduction_image(self, reduc_coeff=0.1):

        # Nettoyage et création du dossier final
        remove_folder_if_exists(self.path_reduced_image)
        
        # On parcourt tous les sous-dossier du dossier raw_image
        # Cela correspond à différentes soirées
        for folder_name in os.listdir(self.path_raw_image):

            input_folder = os.path.join(self.path_raw_image, folder_name)
            output_folder = os.path.join(self.path_reduced_image, folder_name)
            
            # On créer le sous-dossier
            os.makedirs(output_folder)
            
            # Récupère tous les noms de fichiers du sous-dossier raw
            images = [f for f in os.listdir(input_folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in tqdm(images, desc=f"Réduction de {folder_name}"):
                img_path = os.path.join(input_folder, img_name)

                # On ouvre l'image
                img = cv2.imread(img_path)

                shape_img = img.shape

                # Réduction de l'image
                width = int(shape_img[1] * reduc_coeff)
                height = int(shape_img[0] * reduc_coeff)
                dim = (width, height)

                # Redimensionnement
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                
                # Sauvegarde
                out_path = os.path.join(output_folder, img_name)
                cv2.imwrite(out_path, resized_img)

                # Debug
                # print(f"[OK] {img_name} redimensionnée et sauvegardée -> {out_path}")
    



    def detect_and_extract_faces(self, method_face_location="hog", model_face_encoding="large"):

        remove_folder_if_exists(self.path_face_image)

        for folder_name in os.listdir(self.path_reduced_image):
            input_folder = os.path.join(self.path_reduced_image, folder_name)
            output_folder = os.path.join(self.path_face_image, folder_name)
            
            os.makedirs(output_folder)  

            images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            self.faces_data = []
            face_global_id = 0  # Compteur pour chaque visage crop

            for img_name in tqdm(images, desc=f"Traitement de {folder_name}"):
                img_path = os.path.join(input_folder, img_name)
                
                # Utilise face_recognition pour lire l'image (ou cv2, au choix)
                image = face_recognition.load_image_file(img_path)
                
                # Localisations des visages
                face_locations = face_recognition.face_locations(image, model=method_face_location) 
                # model="hog" est plus rapide que "cnn"
                
                # Embeddings de chaque visage
                face_encodings = face_recognition.face_encodings(image, face_locations, model=model_face_encoding)
                # model="large" est plus précis que "small"
                # “small” (default) returns 5 points but is faster.
                
                for i, (loc, enc) in enumerate(zip(face_locations, face_encodings)):
                    top, right, bottom, left = loc
                    
                    # Crop du visage (en se basant sur face_recognition qui renvoie en (top, right, bottom, left))
                    face_image = image[top:bottom, left:right]
                    
                    # Convertir en format BGR pour l’enregistrement avec cv2
                    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    
                    face_crop_name = f"{os.path.splitext(img_name)[0]}_face{face_global_id}.jpg"
                    face_crop_path = os.path.join(output_folder, face_crop_name)
                    
                    # Sauvegarde du crop
                    cv2.imwrite(face_crop_path, face_image_bgr)
                    
                    # Sauvegarder infos dans la liste
                    self.faces_data.append({
                        "face_id": face_global_id,
                        "embedding": enc,
                        "photo_name": img_name,
                        "crop_path": face_crop_path
                    })
                    
                    face_global_id += 1
                
                # Débug
                # print(f"[OK] {len(face_encodings)} visages détectés dans {img_name}.")
    

    def cluster_faces(self):
        # On récupère tous les embeddings dans un array numpy
        embeddings = np.array([face["embedding"] for face in self.faces_data])

        # Appliquer DBSCAN
        dbscan = DBSCAN(metric='euclidean', eps=0.5, min_samples=2)
        dbscan.fit(embeddings)

        labels = dbscan.labels_  # cluster_id pour chaque embedding (ou -1 si outlier)

        # On rajoute le cluster dans les données de faces_data
        for i, face in enumerate(self.faces_data):
            face["cluster_id"] = labels[i]

        print(f"Nombre de clusters trouvés (hors -1) : {len(set(labels) - {-1})}")


    def visualize_clusters(self, nb_affichage=5):
        
        # Regrouper les visages par cluster_id
        clusters_dict = defaultdict(list)
        for face in self.faces_data:
            cluster_id = face["cluster_id"]
            clusters_dict[cluster_id].append(face["crop_path"])

        # Ignorer le cluster -1 (outliers), si souhaité
        cluster_ids = sorted([cid for cid in clusters_dict.keys() if cid != -1])

        print(f"Clusters ignorés (outliers) : {len(clusters_dict.get(-1, []))}")
        print(f"Clusters valides : {len(cluster_ids)}")

        for cid in cluster_ids:
            face_paths = clusters_dict[cid]
            
            # On ouvre une nouvelle figure pour chaque cluster
            plt.figure(figsize=(10, 2))
            plt.suptitle(f"Cluster {cid}\n{len(face_paths)} visages", fontsize=14)
            
            # Afficher un échantillon de visages (jusqu’à max_faces_per_cluster)
            sample_paths = face_paths[:nb_affichage]
            
            for i, path in enumerate(sample_paths):
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    continue
                # Convertir en RGB pour matplotlib
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                plt.subplot(1, len(sample_paths), i+1)
                plt.imshow(img_rgb)
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()

def remove_folder_if_exists(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)