# 1. 切换到项目目录
E:
cd E:\KKGG

# 2. 测试运行（处理第一篇文章）
python main_cli.py --task entity_disambiguation --data_dir E:\KKGG --output_dir E:\KKGG\output --start_index 0 --end_index 0

# 3. 检查是否成功
echo 检查上面是否有错误信息，如果没有错误则继续

# 4. 继续处理所有文章
python main_cli.py --task entity_disambiguation --data_dir E:\KKGG --output_dir E:\KKGG\output --start_index 0 --end_index -1

# 5. 处理概念聚类
python main_cli.py --task concept_clustering --data_dir E:\KKGG --output_dir E:\KKGG\output --start_index 0 --end_index -1