import numpy as np
import torch
import trimesh
import pickle
import os
import time
import torch.nn as nn
from smplx.lbs import batch_rodrigues, lbs, vertices2landmarks
from smplx.utils import Struct, rot_mat_to_euler
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Funkcje pomocnicze (to_tensor, to_np) ---
def to_tensor(array, dtype=torch.float32, device='cpu'):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    return array.to(device)

def to_np(array, dtype=None):
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    np_array = np.array(array)
    if np_array.dtype == 'object' and np_array.size > 0:
        first_item = np_array.item(0)
        if 'scipy.sparse' in str(type(first_item)):
            np_array = first_item.todense()
    if dtype is not None:
        return np.array(np_array, dtype=dtype)
    else:
        return np.array(np_array)

# --- KLASY FLAME, Masking, SimpleConfig (bez zmian) ---
# ... (Wklej tutaj swoje oryginalne, niezmienione klasy FLAME_PYTORCH, Masking i SimpleConfig)
# --- Ze względu na długość, pomijam je tutaj, ale muszą one znaleźć się w finalnym pliku ---
class FLAME_PYTORCH(nn.Module):
    def __init__(self, config, device):
        super(FLAME_PYTORCH, self).__init__()
        print("Creating the FLAME Decoder")
        self.device = device
        with open(config.flame_model_path, "rb") as f:
            flame_model_data = pickle.load(f, encoding="latin1")
        self.flame_model = Struct(**flame_model_data)
        self.NECK_IDX = 1; self.batch_size = config.batch_size; self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour; self.faces = self.flame_model.f
        self.register_buffer("faces_tensor", to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long, device=self.device))
        default_shape = torch.zeros([self.batch_size, 300 - config.shape_params], dtype=self.dtype,requires_grad=False,).to(self.device)
        self.register_parameter("shape_betas", nn.Parameter(default_shape, requires_grad=False))
        default_exp = torch.zeros([self.batch_size, 100 - config.expression_params],dtype=self.dtype,requires_grad=False,).to(self.device)
        self.register_parameter("expression_betas", nn.Parameter(default_exp, requires_grad=False))
        default_eyball_pose = torch.zeros([self.batch_size, 6], dtype=self.dtype, requires_grad=False).to(self.device)
        self.register_parameter("eye_pose_params", nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False).to(self.device)
        self.register_parameter("neck_pose_params", nn.Parameter(default_neck_pose, requires_grad=False))
        self.use_3D_translation = config.use_3D_translation
        default_transl = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False).to(self.device)
        self.register_parameter("transl_params", nn.Parameter(default_transl, requires_grad=False))
        self.register_buffer("v_template", to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype, device=self.device))
        self.register_buffer("shapedirs", to_tensor(to_np(self.flame_model.shapedirs), dtype=self.dtype, device=self.device))
        self.register_buffer("J_regressor", to_tensor(to_np(self.flame_model.J_regressor), dtype=self.dtype, device=self.device))
        posedirs = np.reshape(self.flame_model.posedirs, [-1, self.flame_model.posedirs.shape[-1]]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype, device=self.device))
        parents = to_tensor(to_np(self.flame_model.kintree_table[0]), dtype=torch.long, device=self.device)
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", to_tensor(to_np(self.flame_model.weights), dtype=self.dtype, device=self.device))
        with open(config.static_landmark_embedding_path, "rb") as f:
            static_embeddings = Struct(**pickle.load(f, encoding="latin1"))
        self.register_buffer("lmk_faces_idx", torch.tensor(static_embeddings.lmk_face_idx.astype(np.int64), dtype=torch.long, device=self.device))
        self.register_buffer("lmk_bary_coords", torch.tensor(static_embeddings.lmk_b_coords, dtype=self.dtype, device=self.device))
        if self.use_face_contour:
            dynamic_embeddings_data = np.load(config.dynamic_landmark_embedding_path, allow_pickle=True, encoding='latin1')[()]
            dynamic_lmk_faces_idx_np = np.array(dynamic_embeddings_data['lmk_face_idx'], dtype=np.int64)
            self.register_buffer("dynamic_lmk_faces_idx", torch.from_numpy(dynamic_lmk_faces_idx_np).to(self.device))
            dynamic_lmk_bary_coords_np = np.array(dynamic_embeddings_data['lmk_b_coords'], dtype=np.float32)
            self.register_buffer("dynamic_lmk_bary_coords", torch.from_numpy(dynamic_lmk_bary_coords_np).to(self.dtype).to(self.device))
            neck_kin_chain = []; curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long, device=self.device)
            while curr_idx != -1: neck_kin_chain.append(curr_idx); curr_idx = self.parents[curr_idx]
            self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))
    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dtype_passed=torch.float32):
        batch_size = pose.shape[0]
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, self.neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        rel_rot_mat = (torch.eye(3, device=self.device, dtype=dtype_passed).unsqueeze_(dim=0).expand(batch_size, -1, -1))
        for idx in range(len(self.neck_kin_chain)): rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
        y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long); mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_lmk_faces_idx = torch.index_select(self.dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(self.dynamic_lmk_bary_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords
    def forward(self, shape_params, expression_params, pose_params, neck_pose=None, eye_pose=None, transl=None):
        batch_size = shape_params.shape[0]
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.shape_betas = torch.zeros([self.batch_size, 300-shape_params.shape[1]], dtype=self.dtype, device=self.device)
            self.expression_betas = torch.zeros([self.batch_size, 100-expression_params.shape[1]], dtype=self.dtype, device=self.device)
            self.eye_pose_params = torch.zeros([self.batch_size, 6], dtype=self.dtype, device=self.device)
            self.neck_pose_params = torch.zeros([self.batch_size, 3], dtype=self.dtype, device=self.device)
            self.transl_params = torch.zeros([self.batch_size, 3], dtype=self.dtype, device=self.device)
        betas = torch.cat([shape_params, self.shape_betas, expression_params, self.expression_betas], dim=1)
        neck_pose_actual = neck_pose if neck_pose is not None else self.neck_pose_params
        eye_pose_actual = eye_pose if eye_pose is not None else self.eye_pose_params
        transl_actual = transl if transl is not None else self.transl_params
        full_pose = torch.cat([pose_params[:, :3], neck_pose_actual, pose_params[:, 3:], eye_pose_actual], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(batch_size, 1, 1)
        vertices, _ = lbs(betas, full_pose, template_vertices, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights)
        lmk_faces_idx_batch = self.lmk_faces_idx.unsqueeze(dim=0).repeat(batch_size, 1)
        lmk_bary_coords_batch = self.lmk_bary_coords.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_b_coords = self._find_dynamic_lmk_idx_and_bcoords(full_pose, dtype_passed=self.dtype)
            lmk_faces_idx_batch = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx_batch], 1)
            lmk_bary_coords_batch = torch.cat([dyn_lmk_b_coords, lmk_bary_coords_batch], 1)
        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx_batch, lmk_bary_coords_batch)
        if self.use_3D_translation:
            landmarks += transl_actual.unsqueeze(dim=1); vertices += transl_actual.unsqueeze(dim=1)
        return vertices, landmarks

class Masking(nn.Module):
    def __init__(self, flame_masks_path, generic_model_path, device):
        super(Masking, self).__init__()
        self.device = device
        if not os.path.exists(flame_masks_path): raise FileNotFoundError(f"FLAME_masks.pkl not found at {flame_masks_path}")
        with open(flame_masks_path, 'rb') as f: self.masks = Struct(**pickle.load(f, encoding='latin1'))
        if not os.path.exists(generic_model_path): raise FileNotFoundError(f"generic_model.pkl not found at {generic_model_path}")
        with open(generic_model_path, 'rb') as f: flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.dtype = torch.float32
        self.register_buffer('faces', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long, device=self.device))
        self.register_buffer('vertices_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype, device=self.device))
    def get_mask_vertices(self, region_name):
        if hasattr(self.masks, region_name): return getattr(self.masks, region_name)
        elif region_name == 'eyes': return np.unique(np.concatenate((self.masks.left_eyeball, self.masks.right_eyeball)))
        elif region_name == 'ears': return np.unique(np.concatenate((self.masks.left_ear, self.masks.right_ear)))
        else: print(f"Warning: Mask region '{region_name}' not found."); return np.array([], dtype=int)

class SimpleConfig:
    def __init__(self, model_dir="./model"):
        self.flame_model_path = os.path.join(model_dir, "generic_model.pkl")
        self.static_landmark_embedding_path = os.path.join(model_dir, "flame_static_embedding.pkl")
        self.dynamic_landmark_embedding_path = os.path.join(model_dir, "flame_dynamic_embedding.npy")
        self.flame_masks_path = os.path.join(model_dir, "FLAME_masks.pkl")
        self.batch_size = 1; self.shape_params = 100; self.expression_params = 50
        self.use_face_contour = True; self.use_3D_translation = False
        for pth_attr in ['flame_model_path', 'static_landmark_embedding_path', 'flame_masks_path']:
            if not os.path.exists(getattr(self, pth_attr)): raise FileNotFoundError(f"Error: File not found: {getattr(self, pth_attr)}")
        if self.use_face_contour and not os.path.exists(self.dynamic_landmark_embedding_path):
             raise FileNotFoundError(f"Error: Dynamic landmark file not found: {self.dynamic_landmark_embedding_path}")

# --- NOWE I POPRAWIONE FUNKCJE ---

def plot_mesh_matplotlib(vertices, faces, output_path, title,
                         highlight_indices=None, highlight_colors='r',
                         other_indices=None, other_colors='b',
                         view_elevation=30, view_azimuth=30,
                         show_full_mesh=True):
    """Rozbudowana funkcja do rysowania siatki i zapisywania jej do pliku."""
    print(f"Generowanie obrazu: {output_path}")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    verts_np = to_np(vertices)
    
    if show_full_mesh and faces is not None:
        ax.plot_trisurf(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2], 
                        triangles=faces, color='lightgrey', alpha=0.5,
                        linewidth=0, antialiased=False)

    if other_indices is not None and len(other_indices) > 0:
        other_points = verts_np[other_indices]
        ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2],
                   s=15, c=other_colors, label='Region of Interest / Rejected', depthshade=False)

    if highlight_indices is not None and len(highlight_indices) > 0:
        highlight_points = verts_np[highlight_indices]
        ax.scatter(highlight_points[:, 0], highlight_points[:, 1], highlight_points[:, 2],
                   s=50, c=highlight_colors, marker='o', edgecolors='k', depthshade=False, label='Selected Points')

    ax.view_init(elev=view_elevation, azim=view_azimuth)

    max_range = np.array([verts_np[:, 0].max()-verts_np[:, 0].min(), 
                          verts_np[:, 1].max()-verts_np[:, 1].min(), 
                          verts_np[:, 2].max()-verts_np[:, 2].min()]).max()
    mid_x = (verts_np[:, 0].max()+verts_np[:, 0].min()) * 0.5
    mid_y = (verts_np[:, 1].max()+verts_np[:, 1].min()) * 0.5
    mid_z = (verts_np[:, 2].max()+verts_np[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.savefig(output_path)
    plt.close(fig)

def find_ear_canal_vertices(vertices_np, masking_util, plot_path_prefix=None):
    """
    (ULEPSZONA WERSJA) Znajduje wierzchołki kanału słuchowego, filtrując punkty leżące
    daleko od centrum ucha, aby uniknąć wyboru punktów na małżowinie.
    """
    ear_regions = {'left': 'left_ear', 'right': 'right_ear'}
    final_indices = {}
    
    for side, mask_name in ear_regions.items():
        print(f"\nPrzetwarzanie ucha: {side.capitalize()}")
        all_mask_indices = masking_util.get_mask_vertices(mask_name)
        if len(all_mask_indices) == 0:
            print(f"Błąd: Maska dla '{mask_name}' jest pusta.")
            return None, None
            
        all_mask_verts = vertices_np[all_mask_indices]
        centroid = np.mean(all_mask_verts, axis=0)
        
        distances_sq_yz = (all_mask_verts[:, 1] - centroid[1])**2 + (all_mask_verts[:, 2] - centroid[2])**2
        
        num_to_keep = max(10, len(all_mask_indices) // 2) 
        sorted_indices_by_dist = np.argsort(distances_sq_yz)
        
        roi_indices_local = sorted_indices_by_dist[:num_to_keep]
        roi_indices_global = all_mask_indices[roi_indices_local]
        
        rejected_indices_local = sorted_indices_by_dist[num_to_keep:]
        rejected_indices_global = all_mask_indices[rejected_indices_local]

        roi_verts = vertices_np[roi_indices_global]
        if side == 'left':
            best_local_idx = np.argmin(roi_verts[:, 0])
        else:
            best_local_idx = np.argmax(roi_verts[:, 0])
            
        final_canal_idx = roi_indices_global[best_local_idx]
        final_indices[side] = final_canal_idx

        print(f"  Finalnie wybrany wierzchołek kanału (indeks): {final_canal_idx}, współrzędne: {vertices_np[final_canal_idx]}")

        if plot_path_prefix:
            plot_path = f"{plot_path_prefix}_{side}_ear_selection.png"
            azimuth = 90 if side == 'left' else -90
            plot_mesh_matplotlib(vertices_np, None, plot_path,
                                 title=f"Wybór punktu w uchu ({side.capitalize()})",
                                 highlight_indices=[final_canal_idx], highlight_colors='red',
                                 other_indices=np.concatenate([roi_indices_global, rejected_indices_global]),
                                 other_colors=['blue']*len(roi_indices_global) + ['lightgrey']*len(rejected_indices_global),
                                 view_elevation=0, view_azimuth=azimuth,
                                 show_full_mesh=False)

    return final_indices.get('left'), final_indices.get('right')


def find_nose_tip_vertex(vertices_np, landmarks_np):
    """Znajduje wierzchołek siatki najbliższy landmarkowi czubka nosa."""
    NOSE_TIP_LANDMARK_INDEX = 30
    if landmarks_np.shape[0] <= NOSE_TIP_LANDMARK_INDEX:
        print("Błąd: Niewystarczająca liczba landmarków do znalezienia nosa.")
        return None
    nose_tip_coord = landmarks_np[NOSE_TIP_LANDMARK_INDEX]
    distances = np.linalg.norm(vertices_np - nose_tip_coord, axis=1)
    return np.argmin(distances)

def translate_vertices(vertices, vector):
    return vertices + vector

def rotate_vertices(vertices, rotation_matrix, center):
    return (vertices - center) @ rotation_matrix.T + center

def seal_mesh_by_triangulating_holes(mesh):
    """
    (WERSJA FINALNA) Uszczelnia siatkę poprzez znalezienie granic, a następnie dla każdej dziury:
    1. Oblicza jej najlepiej dopasowaną płaszczyznę (origin + normal).
    2. Używa trimesh.geometry.plane_transform do rzutowania na 2D.
    3. Trianguluje w 2D.
    4. Transformuje łatę z powrotem do 3D.
    """
    # Importujemy potrzebne, niskopoziomowe funkcje, aby mieć pewność, że są dostępne
    from trimesh.geometry import plane_transform
    import trimesh.transformations as tf
    from trimesh.creation import triangulate_polygon

    if mesh.is_watertight:
        print("\nSiatka jest już szczelna. Pomijanie uszczelniania.")
        return mesh

    print("\nSiatka nie jest szczelna. Rozpoczęcie zaawansowanego uszczelniania...")

    # mesh.outline() zwraca obiekt Path3D, który zawiera pętle krawędzi (dziury)
    # jako listę w atrybucie .entities
    outline = mesh.outline()
    
    # Sprawdzenie, czy w ogóle znaleziono jakieś dziury.
    if not hasattr(outline, 'entities') or len(outline.entities) == 0:
        print("Nie znaleziono granic (entities) do załatania. Próba z prostszym fill_holes().")
        mesh.fill_holes()
        if mesh.is_watertight: 
            print("Sukces, siatka uszczelniona przez fill_holes().")
        else:
            print("Ostrzeżenie: fill_holes() również nie uszczelniło siatki.")
        return mesh

    hole_entities = outline.entities
    all_patches = []
    print(f"Znaleziono {len(hole_entities)} potencjalnych dziur (encji) do załatania.")

    for i, entity in enumerate(hole_entities):
        # Dziura musi być zamkniętą pętlą z co najmniej 3 wierzchołkami, aby utworzyć trójkąt.
        if not entity.closed or len(entity.nodes) < 3:
            print(f"  > Pomijanie encji nr {i+1}, która nie jest zamkniętą pętlą z co najmniej 3 wierzchołkami.")
            continue
            
        # entity.points zwraca INDEKSY wierzchołków z oryginalnej siatki
        boundary_verts_indices = entity.points
        print(f"  > Próba załatania dziury nr {i+1} składającej się z {len(boundary_verts_indices)} wierzchołków...")

        try:
            # Pobieramy współrzędne 3D tych wierzchołków
            verts_3D = mesh.vertices[boundary_verts_indices]
            
            # KROK 1: Obliczenie najlepiej pasującej płaszczyzny (origin i normal) dla dziury
            origin = np.mean(verts_3D, axis=0)
            
            # Najlepsza metoda na znalezienie wektora normalnego to PCA (analiza głównych składowych)
            # na scentrowanych wierzchołkach. Wektor własny odpowiadający najmniejszej wartości własnej
            # będzie normalną do płaszczyzny.
            _, _, vh = np.linalg.svd(verts_3D - origin)
            normal = vh[-1]

            # KROK 2: Uzyskanie macierzy transformacji rzutującej na płaszczyznę 2D i jej odwrotności
            to_2D = plane_transform(origin, normal)
            to_3D = np.linalg.inv(to_2D)
            
            # Rzutujemy wierzchołki pętli do 2D (interesują nas tylko kolumny 0 i 1, czyli X i Y)
            verts_2D = tf.transform_points(verts_3D, to_2D)[:, :2]

            # KROK 3: Triangulacja rzutowanych wierzchołków 2D.
            # 'triangle' to solidny i zalecany silnik do triangulacji.
            patch_faces_local, patch_verts_2D = triangulate_polygon(verts_2D, engine='triangle')
            
            # KROK 4: Rzutowanie wierzchołków łaty z powrotem do 3D
            # Tworzymy jednorodne współrzędne 3D (z Z=0), zanim zastosujemy transformację powrotną.
            patch_verts_3D_homo = np.column_stack([
                patch_verts_2D,
                np.zeros(len(patch_verts_2D))
            ])
            patch_verts_3D = tf.transform_points(patch_verts_3D_homo, to_3D)

            # KROK 5: Utworzenie siatki dla samej łaty i dodanie jej do listy
            patch_mesh = trimesh.Trimesh(vertices=patch_verts_3D, faces=patch_faces_local, process=False)
            all_patches.append(patch_mesh)
            
            print(f"    Pomyślnie utworzono łatę z {len(patch_faces_local)} trójkątów.")

        except Exception as e:
            # Łapanie wyjątków jest ważne, bo triangulacja może się nie udać dla zdegenerowanych kształtów.
            print(f"    Ostrzeżenie: Nie udało się załatać dziury nr {i+1}. Błąd: {e}")

    if not all_patches:
        print("Nie udało się utworzyć żadnych łat. Zwracam oryginalny model.")
        return mesh
        
    # KROK 6: Połączenie oryginalnej siatki ze wszystkimi nowymi łatami w jeden obiekt
    print("Łączenie oryginalnej siatki z wygenerowanymi łatami...")
    
    final_mesh = trimesh.util.concatenate([mesh] + all_patches)
    # process=True jest ważne, aby m.in. połączyć zduplikowane wierzchołki na granicach
    final_mesh.process(validate=True)
    
    if final_mesh.is_watertight:
        print("Sukces! Siatka została pomyślnie uszczelniona.")
    else:
        # Czasami po konkatenacji zostają mikroskopijne dziurki lub problemy z orientacją normalnych.
        print("Siatka wciąż nie jest szczelna. Próba użycia fill_holes() i fix_normals() na drobnych pozostałościach...")
        final_mesh.fill_holes()
        final_mesh.fix_normals() # Naprawia orientację trójkątów
        if final_mesh.is_watertight:
            print("Sukces po drugiej próbie! Siatka jest teraz szczelna.")
        else:
             print("Ostrzeżenie: Siatka nadal nie jest szczelna po wszystkich próbach.")

    return final_mesh

# --- GŁÓWNA LOGIKA PROGRAMU ---
if __name__ == "__main__":
    seed = int(time.time())
    # seed = 1715873111 # Możesz ustawić konkretne ziarno do testów
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Używane ziarno losowania (seed): {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    config = SimpleConfig(model_dir="./model")
    flamelayer = FLAME_PYTORCH(config, device)
    masking_util = Masking(config.flame_masks_path, config.flame_model_path, device)

    output_models_dir = "output_models"
    output_plots_dir = "output_plots"
    os.makedirs(output_models_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)
    run_id = f"head_seed_{seed}"

    # --- 1. Generowanie losowego modelu ---
    print("\n--- Etap 1: Generowanie modelu ---")
    shape_params = torch.randn(1, config.shape_params, device=device) * 1.5
    expression_params = torch.zeros(1, config.expression_params, device=device)
    pose_params = torch.zeros(1, 6, device=device)
    
    with torch.no_grad():
        verts_orig_torch, landmarks_torch = flamelayer(shape_params, expression_params, pose_params)
    
    vertices_current = to_np(verts_orig_torch[0])
    landmarks_np = to_np(landmarks_torch[0])
    faces_np = to_np(flamelayer.faces_tensor)
    
    trimesh.Trimesh(vertices=vertices_current, faces=faces_np).export(os.path.join(output_models_dir, f"{run_id}_0_original.obj"))
    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_0_original.png"), "Oryginalny model")

    # --- 2. Znajdowanie punktów kluczowych ---
    print("\n--- Etap 2: Znajdowanie punktów kluczowych ---")
    idx_left_canal, idx_right_canal = find_ear_canal_vertices(vertices_current, masking_util, plot_path_prefix=os.path.join(output_plots_dir, run_id))
    idx_nose_tip = find_nose_tip_vertex(vertices_current, landmarks_np)

    if any(idx is None for idx in [idx_left_canal, idx_right_canal, idx_nose_tip]):
        print("\nNie udało się znaleźć wszystkich punktów kluczowych. Przerwanie przetwarzania.")
        exit()

    key_points_indices = [idx_left_canal, idx_right_canal, idx_nose_tip]
    key_points_colors = ['red', 'blue', 'green'] # L, R, Nose
    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_1_key_points.png"), 
                         "Oryginalny model z punktami kluczowymi",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)

    # --- 3. Wyrównywanie modelu (krok po kroku) ---
    print("\n--- Etap 3: Wyrównywanie modelu ---")
    
    # Krok 3.1: Wstępne obroty dla lepszej orientacji
    print("\nKrok 3.1: Wstępne obroty")
    rot1 = R_scipy.from_euler('x', 90, degrees=True).as_matrix()
    vertices_current = rotate_vertices(vertices_current, rot1, center=[0,0,0])
    rot2 = R_scipy.from_euler('z', 90, degrees=True).as_matrix()
    vertices_current = rotate_vertices(vertices_current, rot2, center=[0,0,0])
    
    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_2_after_initial_rot.png"), 
                         "Po wstępnych obrotach",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)

    # Krok 3.2: Wyrównanie osi uszu do osi Y
    print("\nKrok 3.2: Wyrównanie osi uszu do osi Y")
    v_left = vertices_current[idx_left_canal]
    v_right = vertices_current[idx_right_canal]
    ear_midpoint = (v_left + v_right) / 2.0
    ear_vector = v_left - v_right # Wektor od prawego do lewego ucha
    target_vector_y = np.array([0, 1.0, 0])
    
    rot3, _ = R_scipy.align_vectors([target_vector_y], [ear_vector])
    vertices_current = rotate_vertices(vertices_current, rot3.as_matrix(), center=ear_midpoint)
    
    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_3_ears_aligned_to_Y.png"), 
                         "Oś uszu równoległa do Y",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)

    # Krok 3.3: Przesunięcie środka linii uszu do płaszczyzny YZ (X=0, Z=0)
    print("\nKrok 3.3: Wycentrowanie linii uszu")
    v_left_curr = vertices_current[idx_left_canal]
    v_right_curr = vertices_current[idx_right_canal]
    ear_midpoint_curr = (v_left_curr + v_right_curr) / 2.0
    translation_vec1 = -ear_midpoint_curr
    vertices_current = translate_vertices(vertices_current, translation_vec1)

    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_4_ears_centered.png"), 
                         "Linia uszu wycentrowana w (0,0,0)",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)
    
    # Krok 3.4: Obrót wokół osi Y, aby nos znalazł się w płaszczyźnie XY (Z=0)
    print("\nKrok 3.4: Obrót, by nos znalazł się w płaszczyźnie XY")
    v_nose_curr = vertices_current[idx_nose_tip]
    angle_rad = np.arctan2(v_nose_curr[2], v_nose_curr[0]) # Kąt w płaszczyźnie XZ
    # Chcemy obrócić o -angle_rad, aby Z stało się zerem
    rot4 = R_scipy.from_euler('y', -np.rad2deg(angle_rad), degrees=True).as_matrix()
    vertices_final = rotate_vertices(vertices_current, rot4, center=[0, 0, 0])

    plot_mesh_matplotlib(vertices_final, faces_np, os.path.join(output_plots_dir, f"{run_id}_5_final_alignment.png"), 
                         "Model finalnie wyrównany",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)
    
    # --- 4. Finalizacja i uszczelnianie ---
    print("\n--- Etap 4: Finalizacja modelu ---")
    
    final_mesh_unsealed = trimesh.Trimesh(vertices=vertices_final, faces=faces_np)
    
    # Krok 4.1: Uszczelnianie modelu za pomocą nowej funkcji
    final_mesh_watertight = seal_mesh_by_triangulating_holes(final_mesh_unsealed)

    path_final_model = os.path.join(output_models_dir, f"{run_id}_prepared_watertight.obj")
    final_mesh_watertight.export(path_final_model)
    print(f"\nZapisano przygotowany i uszczelniony model do: {path_final_model}")

    # Ostatnia wizualizacja
    plot_mesh_matplotlib(final_mesh_watertight.vertices, final_mesh_watertight.faces, 
                         os.path.join(output_plots_dir, f"{run_id}_6_final_watertight.png"), 
                         "Finalny, uszczelniony model")

    print("\nPrzetwarzanie zakończone pomyślnie.")