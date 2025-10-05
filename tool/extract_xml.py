import os
import re
import bz2
from lxml import etree
from tqdm import tqdm

def clean_wiki_text(text):

    if not text:
        return ""
    

    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    

    for _ in range(10):
        new_text = re.sub(r'{{(?:[^{}]|{[^{}]*})*}}', '', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
    

    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    

    text = re.sub(r'<[^>]+>', '', text)
    

    text = re.sub(r'{\|.*?\|}', '', text, flags=re.DOTALL)
    text = re.sub(r'<gallery>.*?</gallery>', '', text, flags=re.DOTALL)
    

    text = re.sub(
        r'[^\u4e00-\u9fff，。！？；："“”‘’（）《》【】、\s]', 
        '', 
        text
    )
    
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)  
    text = re.sub(r'( ?\n ?)+', '\n', text)     
    

    lines = [line.strip() for line in text.split('\n') 
             if len(line.strip()) >= 5]
    
    return '\n'.join(lines)

def process_single_xml(input_path, output_path):
    # 支持bz2压缩文件直接读取
    if input_path.endswith('.bz2'):
        open_func = bz2.open
        open_mode = 'rb'
    else:
        open_func = open
        open_mode = 'rb'  
    
    page_count = 0
    with open_func(input_path, open_mode) as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        context = etree.iterparse(infile, events=('end',), tag='{*}page')
        
        for event, elem in tqdm(context, desc=f"处理 {os.path.basename(input_path)}"):

            ns = elem.findtext('{*}ns', default='')
            title = elem.findtext('{*}title', default='')
            redirect = elem.find('{*}redirect')
            
            revision = elem.find('.//{*}revision')
            if revision is not None:
                text_elem = revision.find('{*}text')
                text = text_elem.text if text_elem is not None else ''
            else:
                text = ''
  
            if ns == '0' and redirect is None:
                cleaned = clean_wiki_text(text)
                if cleaned:
                    outfile.write(f"【{title}】\n")
                    outfile.write(cleaned + "\n\n")
                    page_count += 1
            
            elem.clear()

            while elem.getprevious() is not None:
                del elem.getparent()[0]
        
    return page_count

def process_xml_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持.xml和.xml.bz2文件
    xml_files = [f for f in os.listdir(input_dir) 
                if f.endswith(('.xml', '.xml.bz2'))]
    
    total_pages = 0
    
    for xml_file in xml_files:
        input_path = os.path.join(input_dir, xml_file)
        output_path = os.path.join(output_dir, 
                                  f"cleaned_{os.path.splitext(os.path.splitext(xml_file)[0])[0]}.txt")

        print(f"处理文件: {xml_file}")
        page_count = process_single_xml(input_path, output_path)
        print(f"已处理页面数: {page_count}")
        total_pages += page_count
    
    print(f"\n处理完成! 总页面数: {total_pages}")

if __name__ == "__main__":
    input_dir = "xml"       # XML文件所在目录
    output_dir = "big_text"     # 清洗后文本输出目录
    
    process_xml_files(input_dir, output_dir)