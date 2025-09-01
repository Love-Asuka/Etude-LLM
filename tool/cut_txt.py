import os
import shutil

def split_and_move_files(big_text_dir='big_text', text_dir='text', chunk_size=2048*2048):

    # 确保目标文件夹存在
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    
    # 遍历big_text文件夹中的所有文件
    for filename in os.listdir(big_text_dir):
        file_path = os.path.join(big_text_dir, filename)
        
        # 如果是文件而不是目录
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as f:
                    part_num = 1
                    while True:
                        # 读取块数据
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        
                        # 创建分割后的文件名
                        base_name, ext = os.path.splitext(filename)
                        new_filename = f"{base_name}_part{part_num}{ext}"
                        new_file_path = os.path.join(text_dir, new_filename)
                        
                        # 写入分割后的文件
                        with open(new_file_path, 'wb') as chunk_file:
                            chunk_file.write(chunk)
                        
                        part_num += 1
                
                print(f"成功分割文件: {filename} -> {part_num-1}个部分")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    split_and_move_files()
