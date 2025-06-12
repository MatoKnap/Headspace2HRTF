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
    
    if show_full_mesh:
        ax.scatter(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2], 
                   s=1, c='darkgrey', alpha=0.3, label='_nolegend_')

    if other_indices is not None and len(other_indices) > 0:
        other_points = verts_np[other_indices]
        ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2],
                   s=15, c=other_colors, label='Region of Interest / Rejected')

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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.savefig(output_path)
    plt.close(fig)

def find_ear_canal_vertices(vertices_np, masking_util, plot_path_prefix=None):
    """
    Znajduje wierzchołki kanału słuchowego, filtrując punkty leżące daleko od centrum ucha.
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
        sorted_indices_by_dist = np.argsort(distances_sq_yz)
        num_to_keep = len(all_mask_indices) // 2
        
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
                                 view_elevation=80, view_azimuth=azimuth - 45,
                                 show_full_mesh=False)

    return final_indices.get('left'), final_indices.get('right')


def find_nose_tip_vertex(vertices_np, landmarks_np):
    NOSE_TIP_LANDMARK_INDEX = 30
    if landmarks_np.shape[0] <= NOSE_TIP_LANDMARK_INDEX:
        return None
    nose_tip_coord = landmarks_np[NOSE_TIP_LANDMARK_INDEX]
    distances = np.linalg.norm(vertices_np - nose_tip_coord, axis=1)
    return np.argmin(distances)

def translate_vertices(vertices, vector):
    return vertices + vector

def rotate_vertices(vertices, rotation_matrix, center):
    return (vertices - center) @ rotation_matrix.T + center

def scale_model(vertices_np, scale_factor):
    return vertices_np * scale_factor

def make_watertight(mesh):
    if mesh.is_watertight:
        print("\nSiatka jest już szczelna.")
        return mesh
    print("\nSiatka nie jest szczelna. Próba wypełnienia dziur...")
    mesh.fill_holes()
    # print(trimesh.repair.stitch(mesh, faces=None, insert_vertices=True))
    if mesh.is_watertight:
        print("Udało się uszczelnić siatkę.")
    else:
        print("Ostrzeżenie: Siatka nadal nie jest szczelna po próbie naprawy.")
    return mesh

def seal_neck_hole(mesh, masking_util):
    """
    Uszczelnia duży otwór na szyi modelu FLAME.
    1. Znajduje krawędź otworu za pomocą maski 'boundary'.
    2. Tworzy "zatyczkę" z trójkątów w formie wachlarza.
    3. Łączy nową geometrię z istniejącą siatką.
    """
    print("-> Rozpoczynanie procedury uszczelniania szyi...")
    
    # Sprawdzenie, czy siatka w ogóle ma dziury
    if mesh.is_watertight:
        print("   Siatka jest już szczelna. Pomijanie uszczelniania szyi.")
        return mesh

    # 1. Znajdź wierzchołki na granicy szyi
    neck_boundary_indices = masking_util.get_mask_vertices('boundary')
    if len(neck_boundary_indices) == 0:
        print("   Ostrzeżenie: Nie znaleziono wierzchołków dla maski 'boundary'.")
        return mesh

    # 2. Użyj trimesh.outline, aby znaleźć uporządkowaną pętlę krawędzi
    #    To jest bardziej niezawodne niż ręczne sortowanie wierzchołków.
    try:
        outlines = mesh.outline()
        # Znajdźmy pętlę, która w większości składa się z naszych wierzchołków z maski
        neck_loop_idx = -1
        max_overlap = 0
        
        # outlines.entities to lista ścieżek; dla dziur będą to obiekty Path3D
        # których wierzchołki (nodes) tworzą zamkniętą pętlę.
        for i, path in enumerate(outlines.entities):
            # Używamy set do szybkiego sprawdzenia przecięcia
            path_vertices_set = set(path.vertices)
            neck_boundary_set = set(neck_boundary_indices)
            overlap = len(path_vertices_set.intersection(neck_boundary_set))
            if overlap > max_overlap:
                max_overlap = overlap
                neck_loop_idx = i

        if neck_loop_idx == -1:
            raise ValueError("Nie udało się zidentyfikować pętli krawędzi szyi.")

        ordered_indices = outlines.entities[neck_loop_idx].vertices
        print(f"   Znaleziono pętlę krawędzi szyi składającą się z {len(ordered_indices)} wierzchołków.")

    except Exception as e:
        print(f"   Błąd podczas znajdowania krawędzi szyi: {e}. Próba uszczelnienia nieudana.")
        return mesh

    # 3. Oblicz centroid pętli
    boundary_verts = mesh.vertices[ordered_indices]
    center_point = np.mean(boundary_verts, axis=0)

    # 4. Dodaj nowy wierzchołek (centrum zatyczki)
    new_vertices = np.vstack([mesh.vertices, center_point])
    center_idx = len(mesh.vertices) 

    # 5. Stwórz nowe trójkąty (wachlarz)
    new_faces = []
    num_boundary_verts = len(ordered_indices)
    for i in range(num_boundary_verts):
        v1_idx = ordered_indices[i]
        v2_idx = ordered_indices[(i + 1) % num_boundary_verts] # % zapewnia zapętlenie
        new_faces.append([center_idx, v1_idx, v2_idx])
    
    new_faces_np = np.array(new_faces)

    # 6. Połącz geometrię
    combined_faces = np.vstack([mesh.faces, new_faces_np])
    
    # Tworzymy nową siatkę
    sealed_mesh = trimesh.Trimesh(vertices=new_vertices, faces=combined_faces)
    
    # 7. Napraw orientację normalnych
    # Jest to kluczowe po dodaniu nowej geometrii
    sealed_mesh.fix_normals()
    
    print(f"   Szyja została uszczelniona. Nowa siatka ma {len(sealed_mesh.vertices)} wierzchołków i {len(sealed_mesh.faces)} ścian.")
    
    return sealed_mesh

def remove_eyes_and_seal_sockets(mesh, masking_util):
    """
    Usuwa geometrię gałek ocznych i zamyka powstałe po nich oczodoły.
    """
    print("-> Rozpoczynanie procedury usuwania oczu i uszczelniania oczodołów...")

    # 1. Identyfikacja wierzchołków i ścian gałek ocznych
    left_eye_verts_indices = masking_util.get_mask_vertices('left_eyeball')
    right_eye_verts_indices = masking_util.get_mask_vertices('right_eyeball')
    eye_verts_indices = np.union1d(left_eye_verts_indices, right_eye_verts_indices)
    
    # Znajdź ściany, które należą do gałek ocznych (wszystkie 3 wierzchołki ściany muszą być w zestawie)
    face_mask = np.isin(mesh.faces, eye_verts_indices).all(axis=1)
    
    # Usuń ściany gałek ocznych
    mesh.update_faces(~face_mask)
    
    # Usuń nieużywane wierzchołki (w tym te od gałek ocznych)
    mesh.remove_unreferenced_vertices()
    print(f"   Usunięto geometrię gałek ocznych. Pozostało {len(mesh.vertices)} wierzchołków.")
    
    # 2. Teraz siatka ma dwie nowe dziury. Użyjmy tej samej logiki co dla szyi.
    #    mesh.outline() znajdzie wszystkie 3 pętle: szyję i dwa oczodoły.
    
    # Upewnijmy się, że szyja jest już załatana, aby uniknąć pomyłki
    # W praktyce, tę funkcję wywołamy przed załataniem szyi, więc to tylko dla pewności
    
    outlines = mesh.outline(face_indices=np.arange(len(mesh.faces))) # Upewniamy się, że analizuje całą siatkę
    
    if len(outlines.entities) == 0:
        print("   Brak otwartych krawędzi po usunięciu oczu. Model jest już szczelny.")
        return mesh

    print(f"   Znaleziono {len(outlines.entities)} otwartych pętli do załatania (oczekiwane oczodoły + szyja).")
    
    # Nowe wierzchołki i ściany, które będziemy dodawać
    new_verts_to_add = []
    new_faces_to_add = []
    
    current_vertices = mesh.vertices.copy()
    
    for path in outlines.entities:
        ordered_indices = path.vertices
        
        # Pomijamy bardzo duże pętle (prawdopodobnie szyja)
        # Oczodoły mają zazwyczaj 30-60 wierzchołków. Szyja > 100.
        if len(ordered_indices) > 80: 
            print(f"   Pomijam dużą pętlę ({len(ordered_indices)} wierzchołków), zakładając, że to szyja.")
            continue
            
        print(f"   Łatam pętlę o {len(ordered_indices)} wierzchołkach (prawdopodobnie oczodół).")
        
        boundary_verts = current_vertices[ordered_indices]
        center_point = np.mean(boundary_verts, axis=0)
        
        # Dodajemy nowy wierzchołek i zapamiętujemy jego przyszły indeks
        new_verts_to_add.append(center_point)
        center_idx = len(current_vertices) + len(new_verts_to_add) - 1

        # Tworzymy nowe ściany
        num_boundary_verts = len(ordered_indices)
        for i in range(num_boundary_verts):
            v1_idx = ordered_indices[i]
            v2_idx = ordered_indices[(i + 1) % num_boundary_verts]
            new_faces_to_add.append([center_idx, v1_idx, v2_idx])
            
    # Po przejściu przez wszystkie pętle, aktualizujemy siatkę
    if not new_verts_to_add:
        print("   Nie znaleziono odpowiednich pętli do załatania jako oczodoły.")
        return mesh
        
    final_vertices = np.vstack([current_vertices] + new_verts_to_add)
    final_faces = np.vstack([mesh.faces] + new_faces_to_add)
    
    sealed_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
    sealed_mesh.fix_normals()
    
    return sealed_mesh

# --- GŁÓWNA LOGIKA PROGRAMU ---
if __name__ == "__main__":
    seed = int(time.time())
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
    np.save(os.path.join(output_models_dir, f"{run_id}_params.npy"), to_np(shape_params[0]))
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
    
    # Krok 3.1: Wstępne obroty
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
    ear_vector = v_left - v_right
    target_vector_y = np.array([0, 1.0, 0])
    
    rot3, _ = R_scipy.align_vectors([target_vector_y], [ear_vector])
    vertices_current = rotate_vertices(vertices_current, rot3.as_matrix(), center=ear_midpoint)
    
    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_3_ears_aligned_to_Y.png"), 
                         "Oś uszu równoległa do Y",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)

    # Krok 3.3: Przesunięcie linii uszu na oś Y
    print("\nKrok 3.3: Przesunięcie linii uszu na oś Y")
    v_left_curr = vertices_current[idx_left_canal]
    v_right_curr = vertices_current[idx_right_canal]
    ear_midpoint_curr = (v_left_curr + v_right_curr) / 2.0
    translation_vec1 = np.array([-ear_midpoint_curr[0], 0, -ear_midpoint_curr[2]])
    vertices_current = translate_vertices(vertices_current, translation_vec1)

    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_4_ears_on_Y_axis.png"), 
                         "Linia uszu na osi Y (środek w XZ=0)",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)
    
    # Krok 3.4: Obrót w płaszczyźnie XZ, by Z nosa = 0
    print("\nKrok 3.4: Obrót, by Z nosa = 0")
    v_nose_curr = vertices_current[idx_nose_tip]
    angle_rad = np.arctan2(v_nose_curr[2], v_nose_curr[0])
    rot4 = R_scipy.from_euler('y', np.rad2deg(angle_rad), degrees=True).as_matrix()
    vertices_current = rotate_vertices(vertices_current, rot4, center=[0, v_nose_curr[1], 0])

    plot_mesh_matplotlib(vertices_current, faces_np, os.path.join(output_plots_dir, f"{run_id}_5_nose_Z_zeroed.png"), 
                         "Z nosa = 0",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)
    
    # Krok 3.5: Przesunięcie, by Y nosa = 0
    print("\nKrok 3.5: Przesunięcie, by Y nosa = 0")
    v_nose_final = vertices_current[idx_nose_tip]
    translation_vec2 = np.array([0, -v_nose_final[1], 0])
    vertices_final = translate_vertices(vertices_current, translation_vec2)

    plot_mesh_matplotlib(vertices_final, faces_np, os.path.join(output_plots_dir, f"{run_id}_6_final_alignment.png"), 
                         "Model finalnie wyrównany",
                         highlight_indices=key_points_indices, highlight_colors=key_points_colors)
    
    # --- 4. Finalizacja ---
    print("\n--- Etap 4: Finalizacja modelu ---")
    
    # Skalowanie (opcjonalne, zakomentowane)
    # print("\nKrok 4.1: Skalowanie modelu")
    
    print("\nKrok 4.2: Uszczelnianie modelu")
    
    # Tworzymy siatkę trimesh z finalnie zorientowanymi wierzchołkami
    final_mesh = trimesh.Trimesh(vertices=vertices_final, faces=faces_np, process=False)

    # Krok 4.2.1: Usuń oczy i załataj oczodoły
    mesh_no_eyes = remove_eyes_and_seal_sockets(final_mesh, masking_util)
    
    # Zapisz model pośredni do weryfikacji
    path_intermediate_model = os.path.join(output_models_dir, f"{run_id}_intermediate_no_eyes.obj")
    mesh_no_eyes.export(path_intermediate_model)
    print(f"   Zapisano model bez oczu do: {path_intermediate_model}")
    
    # Krok 4.2.2: Załataj dziurę na szyi
    final_mesh_watertight = seal_neck_hole(mesh_no_eyes, masking_util)
    
    # Ostateczne sprawdzenie
    if final_mesh_watertight.is_watertight:
        print("\nSUKCES: Model jest teraz szczelny (watertight)!")
    else:
        # Czasami po operacjach mogą powstać drobne problemy, które standardowe funkcje naprawią
        print("\nOstrzeżenie: Model wciąż nie jest szczelny. Próba automatycznej naprawy za pomocą `trimesh.repair.fill_holes`...")
        final_mesh_watertight.fill_holes()
        if final_mesh_watertight.is_watertight:
            print("SUKCES: Drobne otwory zostały załatane.")
        else:
            broken_face_indices = trimesh.repair.broken_faces(final_mesh_watertight)
            print(f"BŁĄD: Ostateczna siatka wciąż nie jest szczelna. Liczba 'zepsutych' ścian: {len(broken_face_indices)}")

    path_final_model = os.path.join(output_models_dir, f"{run_id}_prepared.obj")
    final_mesh_watertight.export(path_final_model)
    print(f"\nZapisano przygotowany model do: {path_final_model}")

    print("\nPrzetwarzanie zakończone.")