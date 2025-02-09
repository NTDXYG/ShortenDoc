import os
def delete_large_files(directory):
    # 定义 50MB 的字节数
    size_limit = 50 * 1024 * 1024
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, file)
            try:
                # 获取文件大小
                file_size = os.path.getsize(file_path)
                if file_size > size_limit:
                    # 如果文件大小超过 50MB，删除该文件
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}，大小: {file_size / (1024 * 1024):.2f} MB")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
# 指定要处理的目录
directory = "all_dataset"
delete_large_files(directory)