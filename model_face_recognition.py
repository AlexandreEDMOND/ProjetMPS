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
        self.path_raw_image = path_raw_image

        last_folder_name = os.path.basename(os.path.normpath(path_raw_image))

        self.path_reduced_image = os.path.join("data/reduced", last_folder_name)
        self.path_face_image = os.path.join("data/faces", last_folder_name)

        self.max_width = 800
        self.max_height = 600
    

    def reduction_image(self):

        remove_folder_if_exists(self.path_reduced_image)
        
        for folder_name in os.listdir(self.path_raw_image):
            input_folder = os.path.join(self.path_raw_image, folder_name)
            output_folder = os.path.join(self.path_reduced_image, folder_name)
            
            if os.path.isdir(input_folder):
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            
                # Récupère tous les noms de fichiers
                images = [f for f in os.listdir(input_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                for img_name in tqdm(images, desc=f"Réduction de {folder_name}"):
                    img_path = os.path.join(input_folder, img_name)
                    img = cv2.imread(img_path)

                    if img is None:
                        # Si cv2 ne parvient pas à lire l'image, on skip
                        print(f"Impossible de lire {img_path}, on saute ce fichier.")
                        continue

                    h, w = img.shape[:2]
                    
                    # Calcul du ratio de redimensionnement pour garder les proportions
                    scale_w = self.max_width / w
                    scale_h = self.max_height / h
                    scale = min(scale_w, scale_h, 1.0)  # éviter d'agrandir si plus petit que max
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    # Redimensionnement
                    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Sauvegarde
                    out_path = os.path.join(output_folder, img_name)
                    cv2.imwrite(out_path, resized_img)

                    # Debug
                    # print(f"[OK] {img_name} redimensionnée et sauvegardée -> {out_path}")
    
    def detect_and_extract_faces(self):

        if not os.path.exists(self.path_face_image):
            os.makedirs(self.path_face_image)

        for folder_name in os.listdir(self.path_reduced_image):
            input_folder = os.path.join(self.path_reduced_image, folder_name)
            output_folder = os.path.join(self.path_face_image, folder_name)
            
            if os.path.isdir(input_folder):

                if os.path.exists(output_folder):
                    for file in os.listdir(output_folder):
                        file_path = os.path.join(output_folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                else:
                    os.makedirs(output_folder)  

                images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                all_faces_data = []
                face_global_id = 0  # Compteur pour chaque visage crop

                for img_name in images:
                    img_path = os.path.join(input_folder, img_name)
                    
                    # Utilise face_recognition pour lire l'image (ou cv2, au choix)
                    image = face_recognition.load_image_file(img_path)
                    
                    # Localisations des visages
                    face_locations = face_recognition.face_locations(image, model="hog") 
                    # model="hog" est plus rapide que "cnn"
                    
                    # Embeddings de chaque visage
                    face_encodings = face_recognition.face_encodings(image, face_locations, model="large")
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
                        all_faces_data.append({
                            "face_id": face_global_id,
                            "embedding": enc,
                            "photo_name": img_name,
                            "crop_path": face_crop_path
                        })
                        
                        face_global_id += 1
                    
                    print(f"[OK] {len(face_encodings)} visages détectés dans {img_name}.")
    
    def cluster_faces(self, all_faces_data):
        # On récupère tous les embeddings dans un array numpy
        embeddings = np.array([face["embedding"] for face in all_faces_data])

        # Appliquer DBSCAN
        dbscan = DBSCAN(metric='euclidean', eps=0.5, min_samples=2)
        dbscan.fit(embeddings)

        labels = dbscan.labels_  # cluster_id pour chaque embedding (ou -1 si outlier)

        for i, face in enumerate(all_faces_data):
            face["cluster_id"] = labels[i]

        print(f"Nombre de clusters trouvés (hors -1) : {len(set(labels) - {-1})}")


    def visualize_clusters(self, all_faces_data):
        
        # Regrouper les visages par cluster_id
        clusters_dict = defaultdict(list)
        for face in all_faces_data:
            cluster_id = face["cluster_id"]
            clusters_dict[cluster_id].append(face["crop_path"])

        # Ignorer le cluster -1 (outliers), si souhaité
        cluster_ids = sorted([cid for cid in clusters_dict.keys() if cid != -1])

        print(f"Clusters ignorés (outliers) : {clusters_dict.get(-1, [])}")
        print(f"Clusters valides : {cluster_ids}")

        for cid in cluster_ids:
            face_paths = clusters_dict[cid]
            
            # On ouvre une nouvelle figure pour chaque cluster
            plt.figure(figsize=(10, 2))
            plt.suptitle(f"Cluster {cid}\n{len(face_paths)} visages", fontsize=14)
            
            # Afficher un échantillon de visages (jusqu’à max_faces_per_cluster)
            sample_paths = face_paths[:5]
            
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