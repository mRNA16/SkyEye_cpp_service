import os
import cv2
import numpy as np
import sys

def extract_all_frames(video_path, output_dir):
    """提取视频的所有帧并保存为图像文件"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"开始处理视频: {os.path.basename(video_path)}")
    print(f"总帧数: {total_frames}, 帧率: {fps:.2f}FPS")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换颜色空间并调整大小
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        
        # 保存帧为图像文件
        output_file = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_file, frame)
        
        frame_count += 1
        
        # 每处理100帧显示一次进度
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
    
    cap.release()
    print(f"完成处理: {os.path.basename(video_path)}, 共提取 {frame_count} 帧\n")

def preprocess_videos_in_range(video_dir, output_dir, min_id, max_id):
    """处理指定ID范围内的视频文件，提取所有帧"""
    # 获取视频目录下所有的视频文件
    video_extensions = ['.mp4']
    video_files = [f for f in os.listdir(video_dir) 
                  if any(f.endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"在目录 {video_dir} 中未找到视频文件")
        return
    
    # 过滤视频文件，只处理ID在[min_id, max_id]范围内的
    filtered_videos = []
    for video_file in video_files:
        try:
            # 解析文件名格式: X_100.mp4
            file_id = int(video_file.split('_')[0])
            if min_id <= file_id <= max_id:
                filtered_videos.append((file_id, video_file))
        except (ValueError, IndexError):
            # 如果文件名格式不符合预期，跳过该文件
            continue
    
    # 按ID排序
    filtered_videos.sort(key=lambda x: x[0])
    
    print(f"找到 {len(filtered_videos)} 个视频文件在ID范围 [{min_id}, {max_id}] 内")
    if filtered_videos:
        print("视频文件列表:", [f[1] for f in filtered_videos])
    print("-" * 50)
    
    if not filtered_videos:
        print(f"在目录 {video_dir} 中未找到ID在 [{min_id}, {max_id}] 范围内的视频文件")
        return
    
    for file_id, video_file in filtered_videos:
        video_path = os.path.join(video_dir, video_file)
        
        # 创建与视频同名的输出文件夹
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        # 提取所有帧
        extract_all_frames(video_path, video_output_dir)

def main():
    """主函数，处理命令行参数"""
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python process.py <min_id> <max_id>")
        print("示例: python process.py 1 10")
        sys.exit(1)
    
    try:
        min_id = int(sys.argv[1])
        max_id = int(sys.argv[2])
    except ValueError:
        print("错误: 参数必须是整数")
        sys.exit(1)
    
    # 验证参数有效性
    if min_id < 0 or max_id < 0:
        print("错误: ID不能为负数")
        sys.exit(1)
    
    if min_id > max_id:
        print("错误: min_id 不能大于 max_id")
        sys.exit(1)
    
    # 指定视频文件目录和输出目录
    video_dir = r'D:\ftd1fs'
    output_dir = r'D:\ftd1fs\all_frames'
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"开始处理视频文件，ID范围: [{min_id}, {max_id}] (包含边界)")
    print("-" * 50)
    
    preprocess_videos_in_range(video_dir, output_dir, min_id, max_id)
    
    print("所有视频处理完成！")

if __name__ == "__main__":
    main()