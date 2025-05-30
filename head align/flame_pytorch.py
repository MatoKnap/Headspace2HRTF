import numpy as np
import torch
import trimesh
import pickle
import os
import torch.nn as nn
from smplx.lbs import batch_rodrigues, lbs, vertices2landmarks
from smplx.utils import Struct, rot_mat_to_euler
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Użyteczne funkcje pomocnicze (bez zmian) ---
def to_tensor(array, dtype=torch.float32, device='cpu'):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    return array.to(device)

def to_np(array, dtype=None):
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    original_np_array = np.array(array)
    if 'scipy.sparse' in str(type(original_np_array.item(0) if original_np_array.size > 0 else original_np_array)):
        try:
            original_np_array = original_np_array.item(0).todense()
        except:
             if hasattr(original_np_array, 'todense'):
                 original_np_array = original_np_array.todense()
    if dtype is not None:
        return np.array(original_np_array, dtype=dtype)
    else:
        return np.array(original_np_array)

def translate_vertices(vertices, translation_vector):
    return vertices + translation_vector

def rotate_vertices(vertices, rotation_matrix, center_of_rotation=np.array([0,0,0])):
    translated_vertices = vertices - center_of_rotation
    rotated_vertices = translated_vertices @ rotation_matrix.T
    return rotated_vertices + center_of_rotation

# --- Klasa FLAME (bez zmian) ---
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
            try:
                lmk_face_idx_np = np.array(dynamic_embeddings_data['lmk_face_idx'])
                dynamic_lmk_faces_idx_np_casted = lmk_face_idx_np.astype(np.int64)
            except Exception as e: print(f"Error processing dynamic_embeddings_data['lmk_face_idx']: {e}"); raise
            self.register_buffer("dynamic_lmk_faces_idx", torch.tensor(dynamic_lmk_faces_idx_np_casted, dtype=torch.long, device=self.device))
            try:
                lmk_b_coords_np = np.array(dynamic_embeddings_data['lmk_b_coords'])
                dynamic_lmk_bary_coords_np_casted = lmk_b_coords_np.astype(self.dtype.as_numpy_dtype if hasattr(self.dtype, 'as_numpy_dtype') else np.float32)
            except Exception as e: print(f"Error processing dynamic_embeddings_data['lmk_b_coords']: {e}"); raise
            self.register_buffer("dynamic_lmk_bary_coords", torch.tensor(dynamic_lmk_bary_coords_np_casted, dtype=self.dtype, device=self.device))
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
        if batch_size != self.batch_size: # Dynamic batch size update
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

# --- Klasa Masking (bez zmian) ---
class Masking(nn.Module): # ... (identyczna jak poprzednio)
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

# --- Konfiguracja (bez zmian) ---
class SimpleConfig: # ... (identyczna jak poprzednio)
    def __init__(self, model_dir="./model"):
        self.flame_model_path = os.path.join(model_dir, "generic_model.pkl")
        self.static_landmark_embedding_path = os.path.join(model_dir, "flame_static_embedding.pkl")
        self.dynamic_landmark_embedding_path = os.path.join(model_dir, "flame_dynamic_embedding.npy")
        self.flame_masks_path = os.path.join(model_dir, "FLAME_masks.pkl")
        self.batch_size = 1; self.shape_params = 100; self.expression_params = 50
        self.use_face_contour = True; self.use_3D_translation = False
        for pth_attr in ['flame_model_path', 'static_landmark_embedding_path', 'flame_masks_path']:
            if not os.path.exists(getattr(self, pth_attr)): print(f"Error: File not found: {getattr(self, pth_attr)}"); exit()
        if self.use_face_contour and not os.path.exists(self.dynamic_landmark_embedding_path):
             print(f"Error: Dynamic landmark file not found: {self.dynamic_landmark_embedding_path}"); exit()

# --- Wizualizacja Matplotlib (bez zmian) ---
def plot_mesh_matplotlib(vertices, faces, output_path="plot.png", title="Mesh Plot", view_elevation=30, view_azimuth=30, highlight_indices=None, highlight_color='r', highlight_size=20, xlim=None, ylim=None, zlim=None, show_axis=True):
    print(f"Plotting with Matplotlib to: {output_path}")
    fig = plt.figure(figsize=(10, 10)); ax = fig.add_subplot(111, projection='3d'); ax.set_title(title)
    verts_np = to_np(vertices); faces_np_int = to_np(faces, dtype=np.int32)
    ax.scatter(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2], s=0.5, c='darkgrey', alpha=0.6)
    if highlight_indices is not None and len(highlight_indices) > 0:
        hp = verts_np[highlight_indices]; ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], s=highlight_size, c=highlight_color, marker='o', edgecolors='black', depthshade=False)
    ax.view_init(elev=view_elevation, azim=view_azimuth)
    if xlim and ylim and zlim: ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    else:
        max_r = np.array([verts_np[:,i].max()-verts_np[:,i].min() for i in range(3)]).max()*0.6
        mid = [ (verts_np[:,i].max()+verts_np[:,i].min())*0.5 for i in range(3)]
        ax.set_xlim(mid[0]-max_r, mid[0]+max_r); ax.set_ylim(mid[1]-max_r, mid[1]+max_r); ax.set_zlim(mid[2]-max_r, mid[2]+max_r)
    if show_axis: ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    else: ax.set_axis_off()
    plt.savefig(output_path, dpi=150); plt.close(fig); print(f"Matplotlib plot saved to {output_path}")

# --- POPRAWIONA FUNKCJA ZNAJDOWANIA KANAŁÓW USZNYCH ---
def find_ear_canal_vertices_from_mask(vertices_np, masking_util):
    """Znajduje przybliżone wierzchołki kanału słuchowego na podstawie maski.
    Start: Lewe ucho X>0, Prawe ucho X<0.
    """
    left_ear_mask_idxs = masking_util.get_mask_vertices('left_ear')
    right_ear_mask_idxs = masking_util.get_mask_vertices('right_ear')

    if not len(left_ear_mask_idxs) or not len(right_ear_mask_idxs):
        print("Warning: Ear masks not found or empty.")
        return None, None

    left_ear_verts = vertices_np[left_ear_mask_idxs]
    right_ear_verts = vertices_np[right_ear_mask_idxs]

    # Dla lewego ucha (X<0), szukamy wierzchołka z X najbliższym 0 (najmniejsze X)
    idx_left_canal = left_ear_mask_idxs[np.argmin(left_ear_verts[:, 0])]
    # Dla prawego ucha (X>0), szukamy wierzchołka z X najbliższym 0 (największe X)
    idx_right_canal = right_ear_mask_idxs[np.argmax(right_ear_verts[:, 0])]
    
    print(f"Estimated left ear canal (X<0) vertex index: {idx_left_canal}, coords: {vertices_np[idx_left_canal]}")
    print(f"Estimated right ear canal (X>0) vertex index: {idx_right_canal}, coords: {vertices_np[idx_right_canal]}")
    
    # Weryfikacja, czy faktycznie są po oczekiwanych stronach
    if vertices_np[idx_left_canal, 0] <= 0 or vertices_np[idx_right_canal, 0] >= 0:
        print("Warning: Estimated ear canals are not on the expected sides of X-axis. Check model orientation or landmarking.")

    return idx_left_canal, idx_right_canal

def translate_vertices(vertices_np, translation_vector):
    """Przesuwa wszystkie wierzchołki o dany wektor."""
    return vertices_np + translation_vector

def rotate_vertices(vertices_np, rotation_matrix, center_of_rotation):
    """
    Obraca wierzchołki wokół danego centrum rotacji przy użyciu macierzy rotacji.
    vertices_np: (N, 3) tablica wierzchołków.
    rotation_matrix: (3, 3) macierz rotacji.
    center_of_rotation: (3,) wektor centrum rotacji.
    """
    # Przesuń wierzchołki tak, aby centrum rotacji było w origo
    vertices_shifted = vertices_np - center_of_rotation
    
    # Zastosuj rotację. Standardowa konwencja dla R i v jako kolumny to y = R @ v.
    # Dla tablicy wierzchołków (N,3) (wierszowych):
    rotated_shifted_vertices = (rotation_matrix @ vertices_shifted.T).T
    # Alternatywnie i często wydajniej:
    # rotated_shifted_vertices = vertices_shifted @ rotation_matrix.T
    
    # Przesuń wierzchołki z powrotem
    return rotated_shifted_vertices + center_of_rotation

def normalize(v, epsilon=1e-12):
    """Normalizuje wektor."""
    norm = np.linalg.norm(v)
    if norm < epsilon: # Unikaj dzielenia przez zero dla bardzo małych wektorów
        # Można zwrócić wektor zerowy, wektor jednostkowy w jakimś kierunku, lub rzucić błąd
        # Tutaj rzucamy błąd, bo zwykle oznacza to zdegenerowany przypadek
        raise ValueError("Cannot normalize a zero or near-zero vector.")
    return v / norm

def align_head_to_target_orientation(vertices_np, idx_left_ear, idx_right_ear, idx_nose_tip):
    """
    Wyrównuje głowę zgodnie z podanymi kryteriami, używając jednej macierzy transformacji,
    a następnie obraca model o 90 stopni zgodnie z ruchem wskazówek zegara wokół osi Z.

    Kryteria wyrównania (przed końcową rotacją):
    1. Środek interauralny (M) do (0,0,0).
    2. Wierzchołek lewego ucha (V1) na dodatniej osi Y (X=0, Y>0, Z=0).
       (Implikuje to, że prawe ucho V2 będzie na ujemnej osi Y: X=0, Y<0, Z=0).
    3. Wierzchołek nosa (V3) na dodatniej osi X i w płaszczyźnie XY (X>0, Z=0).
       Jego współrzędna Y będzie zerowa tylko jeśli v3_t jest prostopadły do v1_t.
    
    Końcowa rotacja:
    - Obrót o 90 stopni zgodnie z ruchem wskazówek zegara wokół osi Z.
      (stare X -> nowe Y, stare Y -> nowe -X, stare Z -> nowe Z)
    """
    # Użyj zdefiniowanej lokalnie implementacji rotate_vertices
    # W Twoim kodzie, zastąp to swoją globalnie dostępną funkcją rotate_vertices
    # rotate_vertices_impl = rotate_vertices_matmul 

    verts = vertices_np.copy() # Pracujemy na kopii
    
    # Oryginalne współrzędne kluczowych punktów
    v1_orig = verts[idx_left_ear]    # Lewe ucho
    v2_orig = verts[idx_right_ear]   # Prawe ucho
    v3_orig = verts[idx_nose_tip]    # Nos

    print(f"\n--- Rozpoczęcie procesu wyrównywania i rotacji ---")
    print(f"Początkowe Lewe Ucho (V1): {np.array2string(v1_orig, precision=4, floatmode='fixed')}")
    print(f"Początkowe Prawe Ucho (V2): {np.array2string(v2_orig, precision=4, floatmode='fixed')}")
    print(f"Początkowy Nos       (V3): {np.array2string(v3_orig, precision=4, floatmode='fixed')}")

    # Krok 1 (Wyrównanie): Obliczenie wektorów względem środka interauralnego (M)
    interaural_midpoint_M = (v1_orig + v2_orig) / 2.0
    
    v1_t = v1_orig - interaural_midpoint_M # Wektor od M do lewego ucha
    v3_t = v3_orig - interaural_midpoint_M # Wektor od M do nosa
    
    print(f"\nKrok 1 (Wyrównanie): Wektory względem środka interauralnego M={np.array2string(interaural_midpoint_M, precision=4, floatmode='fixed')}")
    print(f"  v1_t (LeweUcho - M): {np.array2string(v1_t, precision=4, floatmode='fixed')}")
    print(f"  v3_t (Nos - M)     : {np.array2string(v3_t, precision=4, floatmode='fixed')}")

    # Krok 2 (Wyrównanie): Definicja osi docelowego układu współrzędnych w przestrzeni modelu
    try:
        model_Y_axis = normalize(v1_t) # Ta oś stanie się globalną +Y po wyrównaniu
    except ValueError:
        print("Błąd: Lewe ucho pokrywa się ze środkiem interauralnym. Nie można zdefiniować osi Y.")
        return vertices_np 

    v3_t_component_on_model_Y = np.dot(v3_t, model_Y_axis) * model_Y_axis
    v3_t_orthogonal_to_model_Y = v3_t - v3_t_component_on_model_Y

    try:
        model_X_axis = normalize(v3_t_orthogonal_to_model_Y) # Ta oś stanie się globalną +X po wyrównaniu
    except ValueError:
        print("Ostrzeżenie: Czubek nosa (v3_t) jest współliniowy z interauralną osią Y (v1_t).")
        print("  Docelowy kierunek X dla nosa jest niejednoznaczny.")
        print("  Wybieranie dowolnej osi X prostopadłej do osi Y.")
        arbitrary_vec = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(np.dot(arbitrary_vec, model_Y_axis)), 1.0): 
            arbitrary_vec = np.array([0.0, 1.0, 0.0])
        
        temp_Z_axis = np.cross(model_Y_axis, arbitrary_vec)
        try:
            model_Z_axis_normalized = normalize(temp_Z_axis)
        except ValueError: 
            arbitrary_vec = np.array([0.0,0.0,1.0]) 
            temp_Z_axis = np.cross(model_Y_axis, arbitrary_vec)
            model_Z_axis_normalized = normalize(temp_Z_axis)

        model_X_axis = np.cross(model_Y_axis, model_Z_axis_normalized)

    model_Z_axis = np.cross(model_X_axis, model_Y_axis) # Ta oś stanie się globalną +Z po wyrównaniu

    print(f"\nKrok 2 (Wyrównanie): Zdefiniowane osie modelu (które staną się globalnymi X,Y,Z po wyrównaniu):")
    print(f"  model_X_axis (docelowe +X): {np.array2string(model_X_axis, precision=4, floatmode='fixed')}")
    print(f"  model_Y_axis (docelowe +Y): {np.array2string(model_Y_axis, precision=4, floatmode='fixed')}")
    print(f"  model_Z_axis (docelowe +Z): {np.array2string(model_Z_axis, precision=4, floatmode='fixed')}")

    # Krok 3 (Wyrównanie): Konstrukcja macierzy rotacji wyrównującej
    alignment_rotation_matrix = np.array([
        model_X_axis,
        model_Y_axis,
        model_Z_axis
    ])

    det_alignment_R = np.linalg.det(alignment_rotation_matrix)
    print(f"\nKrok 3 (Wyrównanie): Skonstruowana macierz rotacji wyrównującej (det={det_alignment_R:.4f}):\n{np.array2string(alignment_rotation_matrix, precision=4, floatmode='fixed')}")
    if not np.isclose(det_alignment_R, 1.0):
        print(f"Ostrzeżenie: Wyznacznik macierzy rotacji wyrównującej wynosi {det_alignment_R:.4f} (powinien być 1.0).")

    # Krok 4 (Wyrównanie): Zastosowanie transformacji wyrównującej
    verts_translated = translate_vertices(verts, -interaural_midpoint_M)
    verts_aligned = rotate_vertices(verts_translated, alignment_rotation_matrix, center_of_rotation=np.array([0.0,0.0,0.0]))

    # Weryfikacja współrzędnych kluczowych punktów PO WYRÓWNANIU (przed końcową rotacją)
    aligned_v1 = verts_aligned[idx_left_ear]
    aligned_v2 = verts_aligned[idx_right_ear]
    aligned_v3 = verts_aligned[idx_nose_tip]

    print(f"\n--- Współrzędne kluczowych punktów PO WYRÓWNANIU (przed końcową rotacją) ---")
    print(f"Wyrównane Lewe Ucho (V1): {np.array2string(aligned_v1, precision=4, floatmode='fixed')} (Cel: X=0, Y>0, Z=0)")
    print(f"Wyrównane Prawe Ucho (V2): {np.array2string(aligned_v2, precision=4, floatmode='fixed')} (Cel: X=0, Y<0, Z=0)")
    print(f"Wyrównany Nos       (V3): {np.array2string(aligned_v3, precision=4, floatmode='fixed')} (Cel: X>0, Z=0; Y wg geometrii)")
    
    aligned_v3_x_theoretical = np.dot(v3_t, model_X_axis)
    aligned_v3_y_theoretical = np.dot(v3_t, model_Y_axis)
    aligned_v3_z_theoretical = np.dot(v3_t, model_Z_axis)
    
    print(f"\nTeoretyczne współrzędne Nosa (V3) po wyrównaniu (na podstawie v3_t i osi modelu):")
    print(f"  X: {aligned_v3_x_theoretical:.4f}")
    print(f"  Y: {aligned_v3_y_theoretical:.4f}")
    print(f"  Z: {aligned_v3_z_theoretical:.4f} (Powinno być bliskie zera)")

    if not np.isclose(aligned_v3[1], 0.0):
        print(f"\nUwaga dotycząca współrzędnej Y Nosa po wyrównaniu ({aligned_v3[1]:.4f}):")
        print(f"  Współrzędna Y nosa (po wyrównaniu, przed końcową rotacją) jest niezerowa, ponieważ wektor v3_t")
        print(f"  (od środka interauralnego do nosa) nie był idealnie prostopadły do osi Y (model_Y_axis).")

    # --- MODYFIKACJA: DODANIE KOŃCOWEJ ROTACJI ---
    # Krok 5: Dodatkowa rotacja o 90 stopni zgodnie z ruchem wskazówek zegara wokół osi Z.
    # Transformacja: (x,y,z) -> (y, -x, z)
    # Macierz rotacji M, dla P_new_wiersz = P_old_wiersz @ M:
    # (1,0,0) -> (0,-1,0)  (stare X na -Y')
    # (0,1,0) -> (1, 0,0)  (stare Y na +X')
    # (0,0,1) -> (0, 0,1)  (stare Z na Z')
    final_rotation_matrix = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    
    print(f"\n--- Krok 5: Zastosowanie końcowej rotacji o 90 stopni zgodnie z ruchem wskazówek zegara wokół osi Z ---")
    print(f"Macierz rotacji końcowej (dla P_new = P_old @ R):\n{np.array2string(final_rotation_matrix, precision=4, floatmode='fixed')}")
    
    verts_final = rotate_vertices(verts_aligned, final_rotation_matrix, center_of_rotation=np.array([0.0,0.0,0.0]))
    # --- KONIEC MODYFIKACJI ---

    # Weryfikacja współrzędnych kluczowych punktów po obu transformacjach
    final_v1_coords = verts_final[idx_left_ear]
    final_v2_coords = verts_final[idx_right_ear]
    final_v3_coords = verts_final[idx_nose_tip]

    # Cele po wyrównaniu: V1(0,Y1,0), V2(0,Y2,0), V3(X3,Y3_geom,0)
    # Po rotacji (x,y,z) -> (y,-x,z):
    # V1(Y1,0,0) -> Cel: X>0, Y=0, Z=0
    # V2(Y2,0,0) -> Cel: X<0, Y=0, Z=0
    # V3(Y3_geom, -X3, 0) -> Cel: Y<0, Z=0; X zgodnie z Y_geom

    print(f"\n--- Końcowe współrzędne kluczowych punktów (po wyrównaniu i rotacji o 90 stopni wokół Z) ---")
    print(f"Końcowe Lewe Ucho (V1): {np.array2string(final_v1_coords, precision=4, floatmode='fixed')} (Cel: X>0, Y=0, Z=0)")
    print(f"Końcowe Prawe Ucho (V2): {np.array2string(final_v2_coords, precision=4, floatmode='fixed')} (Cel: X<0, Y=0, Z=0)")
    print(f"Końcowy Nos       (V3): {np.array2string(final_v3_coords, precision=4, floatmode='fixed')} (Cel: Y<0, Z=0; X zgodnie z pierwotnym przesunięciem Y nosa w układzie wyrównanym)")
    
    # Teoretyczne współrzędne Nosa po końcowej rotacji: (aligned_v3_y_theoretical, -aligned_v3_x_theoretical, aligned_v3_z_theoretical)
    expected_final_v3_x = aligned_v3_y_theoretical
    expected_final_v3_y = -aligned_v3_x_theoretical
    expected_final_v3_z = aligned_v3_z_theoretical 
    
    print(f"\nTeoretyczne końcowe współrzędne Nosa (V3) po wszystkich transformacjach:")
    print(f"  X: {expected_final_v3_x:.4f} (Odpowiada Y nosa po wyrównaniu; niezerowe, jeśli v3_t nie był prostopadły do osi uszu)")
    print(f"  Y: {expected_final_v3_y:.4f} (Odpowiada -X nosa po wyrównaniu; <0 jeśli nos był skierowany do przodu)")
    print(f"  Z: {expected_final_v3_z:.4f} (Powinno być bliskie zera)")

    if not np.isclose(final_v3_coords[0], 0.0): # Jeśli końcowa współrzędna X nosa jest niezerowa
        print(f"\nUwaga dotycząca końcowej współrzędnej X Nosa ({final_v3_coords[0]:.4f}):")
        print(f"  Końcowa współrzędna X nosa jest niezerowa. Ta wartość odpowiada współrzędnej Y nosa")
        print(f"  po kroku wyrównania (przed końcową rotacją). Jest niezerowa, ponieważ wektor od środka")
        print(f"  interneuralnego do nosa (v3_t) nie był idealnie prostopadły do osi uszu (model_Y_axis).")
        print(f"  Jest to typowe dla geometrii głowy.")

    return verts_final


if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")
    config = SimpleConfig(model_dir="./model") 
    flamelayer = FLAME_PYTORCH(config, device); flamelayer.to(device)
    
    shape_params = torch.randn(config.batch_size, config.shape_params, device=device) * 1.5 
    expression_params = torch.zeros(config.batch_size, config.expression_params, device=device)
    # Ustawienie orientacji startowej: nos w -Z, lewe ucho X<0, prawe X>0, góra głowy +Y
    # FLAME domyślnie generuje głowę patrzącą w +Z (lub -Z, zależy od interpretacji).
    # Jeśli generuje nos w +Z, musimy obrócić o 180 stopni wokół Y.
    initial_pose_params = torch.zeros(config.batch_size, 6, device=device)
    # initial_pose_params[:, 1] = np.pi # Obrót o 180 stopni wokół osi Y (yaw)
                                     # Sprawdź, czy to daje nos w -Z
    
    print(f"Generated random shape params (first 5): {shape_params[0, :5]}")
    with torch.no_grad():
        vertices_orig_torch, landmarks_torch = flamelayer(shape_params, expression_params, initial_pose_params)
    
    vertices_np_orig = vertices_orig_torch[0].cpu().numpy()
    landmarks_np = landmarks_torch[0].cpu().numpy() # (68, 3)

    if hasattr(flamelayer, 'faces') and isinstance(flamelayer.faces, np.ndarray):
        faces_np = flamelayer.faces
    else: faces_np = flamelayer.faces_tensor.cpu().numpy()
    
    output_dir_renders = "output_flame_renders_mpl_v3"
    os.makedirs(output_dir_renders, exist_ok=True)
    output_dir_models = "output_flame_aligned_v3"
    os.makedirs(output_dir_models, exist_ok=True)
    
    # --- Definicja granic dla plotów ---
    def get_plot_bounds(verts, margin_factor=0.1):
        bounds = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            min_val, max_val = verts[:,i].min(), verts[:,i].max()
            margin = (max_val - min_val) * margin_factor
            bounds[f'{axis}lim'] = (min_val - margin, max_val + margin)
        # Upewnij się, że zakres jest taki sam dla wszystkich osi dla lepszej percepcji 3D
        ranges = [bounds[f'{ax}lim'][1] - bounds[f'{ax}lim'][0] for ax in ['x','y','z']]
        max_plot_range = np.max(ranges)
        
        final_bounds = {}
        for axis in ['x', 'y', 'z']:
            mid = (bounds[f'{axis}lim'][0] + bounds[f'{axis}lim'][1]) / 2
            final_bounds[f'{axis}lim'] = (mid - max_plot_range/2, mid + max_plot_range/2)
        return final_bounds

    orig_bounds = get_plot_bounds(vertices_np_orig)
    plot_mesh_matplotlib(vertices_np_orig, faces_np, 
                         output_path=os.path.join(output_dir_renders, "0_original_front.png"),
                         title="Original - Front (Approx.)", view_elevation=10, view_azimuth=-90, **orig_bounds)
    plot_mesh_matplotlib(vertices_np_orig, faces_np, 
                         output_path=os.path.join(output_dir_renders, "0_original_side.png"),
                         title="Original - Side", view_elevation=10, view_azimuth=0, **orig_bounds)


    # --- Znajdowanie kluczowych punktów ---
    masking_util = Masking(flame_masks_path=config.flame_masks_path, generic_model_path=config.flame_model_path, device=device)
    idx_left_canal, idx_right_canal = find_ear_canal_vertices_from_mask(vertices_np_orig, masking_util)

    # Użycie landmarku dla czubka nosa (indeks 30 dla 68 landmarków)
    # Upewnij się, że landmarki są w tej samej przestrzeni co wierzchołki!
    NOSE_TIP_LANDMARK_INDEX = 30 
    if landmarks_np.shape[0] > NOSE_TIP_LANDMARK_INDEX:
        nose_tip_co_from_landmark = landmarks_np[NOSE_TIP_LANDMARK_INDEX]
        # Znajdź najbliższy wierzchołek siatki do tego landmarku
        distances_to_nose_lmk = np.linalg.norm(vertices_np_orig - nose_tip_co_from_landmark, axis=1)
        idx_nose_tip_vertex = np.argmin(distances_to_nose_lmk)
        print(f"Nose tip landmark (idx {NOSE_TIP_LANDMARK_INDEX}): {nose_tip_co_from_landmark}")
        print(f"Closest mesh vertex to nose tip landmark (index {idx_nose_tip_vertex}): {vertices_np_orig[idx_nose_tip_vertex]}")
    else:
        print(f"Warning: Not enough landmarks ({landmarks_np.shape[0]}) to get nose tip at index {NOSE_TIP_LANDMARK_INDEX}.")
        idx_nose_tip_vertex = None


    if idx_left_canal is None or idx_right_canal is None or idx_nose_tip_vertex is None:
        print("Could not determine all key points (ears, nose). Exiting alignment.")
        # ... zapisz niezmodyfikowany ...
        exit()
    
    keypoints_indices = [idx_left_canal, idx_right_canal, idx_nose_tip_vertex]
    keypoints_colors = ['red', 'blue', 'green'] # L, R, Nose
    plot_mesh_matplotlib(vertices_np_orig, faces_np, 
                         output_path=os.path.join(output_dir_renders, "1_original_keypoints.png"), 
                         title="Original - Keypoints (L:R, R:B, Nose:G)",
                         view_elevation=20, view_azimuth=-70,
                         highlight_indices=keypoints_indices, highlight_color=keypoints_colors, **orig_bounds)

    # --- WYRÓWNANIE ---
    vertices_aligned_np = align_head_to_target_orientation(vertices_np_orig, idx_left_canal, idx_right_canal, idx_nose_tip_vertex)
    
    aligned_bounds = get_plot_bounds(vertices_aligned_np)
    plot_mesh_matplotlib(vertices_aligned_np, faces_np, 
                         output_path=os.path.join(output_dir_renders, "2_aligned_front_X_nose.png"), 
                         title="Aligned - Front (Nose on +X)", view_elevation=10, view_azimuth=-90, # Patrzymy wzdłuż X
                         highlight_indices=keypoints_indices, highlight_color=keypoints_colors, **aligned_bounds)
    plot_mesh_matplotlib(vertices_aligned_np, faces_np, 
                         output_path=os.path.join(output_dir_renders, "2_aligned_side_Y_ears.png"), 
                         title="Aligned - Side (Ears on +Y)", view_elevation=10, view_azimuth=0,   # Patrzymy wzdłuż Y
                         highlight_indices=keypoints_indices, highlight_color=keypoints_colors, **aligned_bounds)
    plot_mesh_matplotlib(vertices_aligned_np, faces_np, 
                         output_path=os.path.join(output_dir_renders, "2_aligned_top_Z.png"), 
                         title="Aligned - Top", view_elevation=90, view_azimuth=-90, # Patrzymy z góry
                         highlight_indices=keypoints_indices, highlight_color=keypoints_colors, **aligned_bounds)

    aligned_mesh_trimesh = trimesh.Trimesh(vertices=vertices_aligned_np, faces=faces_np)
    # Sprawdzenie szczelności (opcjonalne, FLAME powinien być szczelny)
    if not aligned_mesh_trimesh.is_watertight:
        print("Warning: Aligned mesh is not watertight. Attempting to fill holes.")
        aligned_mesh_trimesh.fill_holes()
        if not aligned_mesh_trimesh.is_watertight:
            print("Warning: Could not make the mesh watertight after filling holes.")

    output_path_obj_aligned = os.path.join(output_dir_models, "random_flame_head_aligned_target.obj")
    aligned_mesh_trimesh.export(output_path_obj_aligned)
    print(f"Saved aligned FLAME head to {output_path_obj_aligned}")
    print("Processing complete.")