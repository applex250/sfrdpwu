import os

def get_image_basenames(folder_path, image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    basename_to_filename = {}
    all_basenames = set()
    for filename in os.listdir(folder_path):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            basename, ext = os.path.splitext(filename)
            ext_lower = ext.lower()
            if ext_lower in image_extensions:
                if basename not in basename_to_filename:
                    basename_to_filename[basename] = filename
                all_basenames.add(basename)
    return basename_to_filename, all_basenames

def find_unpaired_images(folder_a, folder_b, image_extensions=None):
    for folder in [folder_a, folder_b]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"文件夹不存在：{folder}")
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"不是有效文件夹：{folder}")
    a_basename_map, a_basenames = get_image_basenames(folder_a, image_extensions)
    b_basename_map, b_basenames = get_image_basenames(folder_b, image_extensions)
    only_in_a = a_basenames - b_basenames
    only_in_b = b_basenames - a_basenames
    only_in_a_files = [a_basename_map[basename] for basename in only_in_a]
    only_in_b_files = [b_basename_map[basename] for basename in only_in_b]
    return {
        "only_in_a": only_in_a_files,
        "only_in_b": only_in_b_files,
        "total_a": len(a_basenames),
        "total_b": len(b_basenames),
        "paired_count": len(a_basenames & b_basenames)
    }

def print_report(result, folder_a, folder_b):
    print("="*60)
    print("图片配对检测报告")
    print("="*60)
    print(f"文件夹A（{folder_a}）有效图片数：{result['total_a']}")
    print(f"文件夹B（{folder_b}）有效图片数：{result['total_b']}")
    print(f"成功配对的图片数：{result['paired_count']}")
    print("-"*60)
    if result['only_in_a']:
        print(f"\n❌ 仅文件夹A存在（{len(result['only_in_a'])}个）：")
        for idx, file in enumerate(result['only_in_a'], 1):
            print(f"  {idx}. {file}")
    else:
        print("\n✅ 文件夹A中无多余图片")
    if result['only_in_b']:
        print(f"\n❌ 仅文件夹B存在（{len(result['only_in_b'])}个）：")
        for idx, file in enumerate(result['only_in_b'], 1):
            print(f"  {idx}. {file}")
    else:
        print("\n✅ 文件夹B中无多余图片")
    print("="*60)

if __name__ == "__main__":
    # ====================== 手动修改这里的路径 ======================
    FOLDER_A = "/home/shared_dir/yao_ce/statehaze1k/Haze1k/Haze1k_moderate/dataset/train/hazy" # 雾图文件夹
    FOLDER_B ="/home/shared_dir/yao_ce/statehaze1k/Haze1k/Haze1k_moderate/dataset/train/GT" # 清晰图文件夹
    # ==============================================================
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    try:
        result = find_unpaired_images(FOLDER_A, FOLDER_B, image_extensions)
        print_report(result, FOLDER_A, FOLDER_B)
    except Exception as e:
        print(f"检测失败：{e}")