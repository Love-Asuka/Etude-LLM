import json
import os
from tkinter import Tk, filedialog, simpledialog

def select_directory(title):

    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def get_pair_count():

    root = Tk()
    root.withdraw()
    count = simpledialog.askinteger(
        "设置", 
        "请输入每个输出文件包含的对话数量:", 
        parent=root,
        minvalue=1,
        initialvalue=1000
    )
    root.destroy()
    return count or 1000 

def process_json_file(input_path, output_folder, conversations_per_file):

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    conversation_count = 0
    file_index = 1
    output_file = None
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                conversations = data.get('conversations', [])
                
                
                if not conversations:
                    continue
                
       
                if conversation_count % conversations_per_file == 0:
                    if output_file:
                        output_file.close()
                    output_path = os.path.join(
                        output_folder, 
                        f"{base_name}_dialogues_{file_index:03d}.jsonl"
                    )
                    output_file = open(output_path, 'w', encoding='utf-8')
                    file_index += 1
                
                output_file.write(
                    json.dumps({"conversations": conversations}, ensure_ascii=False) + '\n'
                )
                conversation_count += 1
                
            except json.JSONDecodeError:
                print(f"警告: 文件 {input_path} 中存在格式错误的JSON行")
    
    if output_file:
        output_file.close()
    
    return conversation_count

def main():
    print("请选择包含JSON文件的输入文件夹:")
    input_folder = select_directory("选择输入文件夹")
    if not input_folder:
        print("未选择输入文件夹，程序退出")
        return
    
    print("请选择输出文件夹:")
    output_folder = select_directory("选择输出文件夹")
    if not output_folder:
        print("未选择输出文件夹，程序退出")
        return
    
    dialogues_per_file = get_pair_count()
    print(f"每个输出文件将包含 {dialogues_per_file} 个完整对话")
    
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(input_folder) 
                 if f.endswith('.json') or f.endswith('.jsonl')]
    
    if not json_files:
        print(f"在文件夹 {input_folder} 中未找到JSON文件")
        return
    
    total_dialogues = 0
    for json_file in json_files:
        input_path = os.path.join(input_folder, json_file)
        print(f"正在处理文件: {json_file}...")
        dialogues_processed = process_json_file(
            input_path, output_folder, dialogues_per_file
        )
        total_dialogues += dialogues_processed
        print(f"  已处理 {dialogues_processed} 个完整对话")
    
    print(f"\n处理完成! 共处理 {len(json_files)} 个文件")
    print(f"生成 {total_dialogues} 个对话样本")
    print(f"输出文件保存在: {output_folder}")

if __name__ == "__main__":
    main()
