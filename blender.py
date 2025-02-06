import argparse
import bpy
import os
import sys
import glob
import shutil
import numpy as np



class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())
    
def is_2d(vertices, tol=1e-6):
    """
    정점 목록(vertices)이 실제 2D 평면에 거의 몰려 있는지 판단합니다.
    PCA를 이용해 공분산 행렬의 고유값을 계산하고,
    가장 작은 고유값과 가장 큰 고유값의 비율이 tol보다 작으면 2D로 간주합니다.
    
    vertices: [(x, y, z), ...] 형태의 리스트
    tol: tolerance (예, 1e-6 이하이면 2D로 판단)
    """
    if not vertices:
        return False

    arr = np.array(vertices)
    arr_centered = arr - np.mean(arr, axis=0)
    cov = np.cov(arr_centered, rowvar=False)
    eigvals, _ = np.linalg.eig(cov)
    min_eig = np.min(eigvals)
    max_eig = np.max(eigvals)
    
    if max_eig == 0:
        return True
    return (min_eig / max_eig) < tol

def save_obj(export_filepath):
    bpy.ops.export_scene.obj(filepath=export_filepath, use_selection=True)




def parse_obj(filename, scale=1.0):

    vertices = []    # (x, y, z)
    texcoords = []   # (u, v)
    faces = []       # (0-indexed)
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertex = tuple(scale * float(num) for num in parts[1:4])
                vertices.append(vertex)
            elif line.startswith('vt '):
                parts = line.split()
                tc = tuple(map(float, parts[1:3]))
                texcoords.append(tc)
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = []
                for part in parts:
                    idx = part.split('/')[0]
                    face.append(int(idx) - 1)
                if len(face) != 3:
                    raise ValueError("현재 코드는 삼각형 면만 지원합니다. 면이 3개 이상의 정점을 갖습니다.")
                faces.append(tuple(face))
    return vertices, texcoords, faces

def main():
    parser = ArgumentParserForBlender()

    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='directory where partnet mobility dataset located.'
    )

    # --obj_idx 인자 추가 (필수)
    parser.add_argument(
        '--obj_idx',
        type=int,
        required=True,
        help='integer instance index of the partnet mobility object, -1 for all objects'
    )

    # 인자 파싱
    args = parser.parse_args()

    # 입력된 인자 출력 (예시)
    print("Dataset Directory: {}".format(args.dataset_dir))
    print("Object Index: {}".format(args.obj_idx))
    
    dataset_dir = args.dataset_dir
    _obj_idx = args.obj_idx
    
    if _obj_idx == -1:
        obj_indices = glob.glob(os.path.join(dataset_dir, '*'))
        obj_indices = [int(obj) for obj in obj_indices] # assert folders contain only the indices of the instances, as the p-mobility format does. 
    else:
        obj_indices = [_obj_idx]
    
    
    for obj_idx in obj_indices:
        base_dir = os.path.join(dataset_dir, str(obj_idx), 'textured_objs')
        save_dir = os.path.join(dataset_dir, str(obj_idx), 'textured_objs_thickness')
        obj_filepaths = glob.glob(os.path.join(base_dir, '**/*'))
        new_thickness = 0.001
        for obj_filepath in obj_filepaths:
            filename = os.path.basename(obj_filepath)
            assert os.path.isfile(obj_filepath), obj_filepath
            # base_dir으로 시작하는지 확인
            if not obj_filepath.startswith(base_dir):
                continue
            new_filepath = obj_filepath.replace(base_dir, save_dir)
            new_dir = os.path.dirname(new_filepath)
            os.makedirs(new_dir, exist_ok=True)
            if filename.lower().endswith('.obj'):
                vertices, texcoords, faces = parse_obj(obj_filepath)
                if not vertices:
                    print("No vertices are available!")
                    shutil.copy(obj_filepath, new_filepath)
                    continue
                if not is_2d(vertices):
                    print("No 2d! just continue")
                    shutil.copy(obj_filepath, new_filepath)
                    continue

                # OBJ 파일 임포트
                bpy.ops.import_scene.obj(filepath=obj_filepath)
                selected_objects = bpy.context.selected_objects
                if not selected_objects:
                    print("선택된 객체가 없습니다. 임포트에 문제가 있을 수 있습니다.")
                else:
                    obj = selected_objects[0]
                    # bpy.context.view_layer.objects.active = obj
                    bpy.context.scene.objects.active = obj

                    # Edit Mode로 들어가서 노멀 재계산 (외부 방향)
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.mesh.normals_make_consistent(inside=False)
                    bpy.ops.object.mode_set(mode='OBJECT')

                    # Solidify Modifier 추가: 노멀 방향(즉, 현재 노멀과 동일 방향)으로 extrusion
                    solidify_mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
                    solidify_mod.thickness = new_thickness
                    solidify_mod.offset = 1.0         # 1.0이면 노멀 방향으로 extrusion
                    solidify_mod.use_flip_normals = False

                    # Modifier 적용
                    bpy.ops.object.modifier_apply(modifier=solidify_mod.name)

                    # OBJ 내보내기
                    save_obj(new_filepath)
                    print("SAVE OBJ FILE:", new_filepath)
            else:
                shutil.copy(obj_filepath, new_filepath)

if __name__ == '__main__':
    main()