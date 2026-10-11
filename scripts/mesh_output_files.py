"""Expose unchanged OBJ meshes under names understandable in a mesh viewer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def obj_counts(path, face_size):
    vertices=0;faces=0;indices=[]
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        parts=line.split('#',1)[0].split()
        if not parts:continue
        if parts[0]=='v':vertices+=1
        elif parts[0]=='f':
            if len(parts)-1!=face_size:
                raise ValueError(f'{path}: expected {face_size} vertices per face')
            faces+=1
            indices.extend(int(v.split('/')[0]) for v in parts[1:])
    if not vertices or not faces or any(i==0 or abs(i)>vertices for i in indices):
        raise ValueError(f'{path}: empty or invalid OBJ')
    return {'vertices':vertices,'faces':faces,'vertices_per_face':face_size}


def export_mesh_files(pipeline_dir, output_dir):
    """Copy real backend meshes; never triangulate, remesh or invent failed outputs."""
    pipeline=Path(pipeline_dir).resolve();output=Path(output_dir).resolve()
    sources={'FINAL_QUAD.obj':(pipeline/'quad/quad_mesh.obj',4),
             'INPUT_TRIANGLE.obj':(pipeline/'backend_bundle/source_mesh.obj',3)}
    records={}
    # Validate every input before creating any output.
    for name,(source,arity) in sources.items():
        records[name]={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),**obj_counts(source,arity)}
        target=output/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=records[name]['sha256']:
            raise FileExistsError(f'Refusing to overwrite a different viewer mesh: {target}')
    output.mkdir(parents=True,exist_ok=True)
    for name,(source,_) in sources.items():
        if source!=output/name:shutil.copyfile(source,output/name)
    (output/'mesh_files.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    return records


def write_run_guide(run_dir, stage2):
    run=Path(run_dir)
    lines=['# 输出文件说明','',
           '**用 MeshLab 查看最终结果：打开本目录的 `FINAL_QUAD.obj`。**' if stage2 else '本次只运行Stage 1，尚未生成最终quad OBJ。','',
           '| 文件 | 内容 | MeshLab打开 |','|---|---|---|']
    if stage2:
        lines += ['| FINAL_QUAD.obj | 训练后经MIQ/QEx生成的最终四边形网格；每个f记录含4个顶点 | 是 |',
                  '| INPUT_TRIANGLE.obj | 后端实际使用的三角输入，与最终quad处于同一坐标系；每个f记录含3个顶点 | 是 |']
    lines += ['| ckpts/hktex2quad_field.pt / .npz | 完整kernel模型参数，用于程序加载，不是网格 | 否 |',
              '| dense_face_field.pt | 逐三角面的方向/尺寸场和几何数据，不是OBJ | 否 |',
              '| visualizations/ | 方向场与kernel诊断图片 | 图片查看器 |',
              '| configs/ / run_status.json | 实际配置和运行状态 | 文本查看器 |','',
              'INPUT_TRIANGLE使用训练归一化坐标，原三角连接关系保留。FINAL_QUAD是算法输出，不是监督用的参考quad。',
              '原始后端文件仍保存在quad_pipeline/；清晰命名的OBJ是逐字节副本，没有重新三角化。']
    (run/'README_OUTPUT_FILES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_mesh_files(args.pipeline_dir,args.output_dir),indent=2))

if __name__=='__main__':main()
